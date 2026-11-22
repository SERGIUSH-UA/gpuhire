"""Ваги епох трену — у R2 ПО ХОДУ трену, presigned PUT без ключів.

Вклеюється рендером перед раннерами трену (`kraken_train`, `parseq_train`).
Обмеження ті самі, що в `_common.py`: лише stdlib, без імпортів gpurunner, без
``from __future__ import annotations``.

Навіщо: ваги епох досі жили на боксі до самого `fetch`. Бокс Vast, що
самознищився за таймером чи пропав разом із хостом, забирав із собою всі
натреновані епохи, хоча кожна з них була готова годинами раніше.

Як: локальний `gpurunner run … -b vast` кладе на бокс файл
``/workspace/gpurunner/train_out.json`` — presigned PUT на заздалегідь названі
ключі (лог, підсумок, ваги з відомими іменами) і пул «слотів» для файлів, чиї
імена наперед невідомі (``checkpoint_NN-0.xxxx.ckpt`` у kraken). Яке справжнє
ім'я лягло в який слот, записує маніфест, що перезаливається після кожного
файла. Ключів R2 на боксі немає й тут.

🔴 Збій вивантаження НЕ валить трен: лише рядок у лозі. Трен — головне, R2 —
страховка від смерті боксу.

Файла специфікації немає (Kaggle, Modal, вимкнено `-p r2_epochs=false`) —
усі методи нічого не роблять.
"""
import hashlib
import json
import threading
import time
from pathlib import Path

#: Тут бекенд Vast кладе специфікацію (`gpurunner.core.train_out.SPEC_NAME` у
#: `REMOTE_ROOT`). Звірено тестом: шлях на двох боках мусить збігатися.
_TO_SPEC_PATH = "/workspace/gpurunner/train_out.json"


def _to_http_put(url, body, size, timeout=600.0):
    """PUT за presigned-посиланням. ``body`` — шлях до файла або bytes."""
    import urllib.request
    headers = {"Content-Type": "application/octet-stream", "Content-Length": str(size)}
    if isinstance(body, (bytes, bytearray)):
        req = urllib.request.Request(url, data=bytes(body), method="PUT", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", 200)
    else:
        with open(body, "rb") as fh:
            req = urllib.request.Request(url, data=fh, method="PUT", headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = getattr(resp, "status", 200)
    if code >= 300:
        raise RuntimeError(f"HTTP {code}")


def _to_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _TrainOutUploader:
    """Фоновий вивантажувач завершених файлів епох.

    Файл вважається завершеним, коли (а) раннер сам про це сказав (`offer`),
    або (б) у теці під наглядом (`watch`) уже з'явився новіший файл того самого
    шаблону, або (в) розмір і час зміни не мінялись між двома обходами і файл
    старший за ``stable_sec``. Перед заливкою і після неї файл переглядається
    ще раз: змінився під час PUT — заливка не зараховується, файл іде в чергу знову.
    """

    def __init__(self, spec=None, *, put=None, log=None, sleep=None, clock=None,
                 poll_sec=30.0, stable_sec=20.0, retries=5, backoff_sec=5.0,
                 live_every_sec=600.0):
        self.spec = spec or {}
        self.enabled = bool(self.spec.get("named") or self.spec.get("slots"))
        self._put = put or _to_http_put
        self._log = log or (lambda m: print(m, flush=True))
        self._sleep = sleep or time.sleep
        self._clock = clock or time.time
        self.poll_sec = float(poll_sec)
        self.stable_sec = float(stable_sec)
        self.retries = int(retries)
        self.backoff_sec = float(backoff_sec)
        self.live_every_sec = float(live_every_sec)
        self._named = dict(self.spec.get("named") or {})
        self._slots = list(self.spec.get("slots") or [])
        self._slot_of = {}
        self._next_slot = 0
        self._slots_warned = False
        self._watches = []
        self._live = {}
        self._seen = {}
        self._done = {}
        self._files = {}
        self._failed = {}
        self._queue = []
        self._qlock = threading.Lock()
        self._ulock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._finished = False

    # ── що вивантажувати ─────────────────────────────────────────────────────
    def watch(self, directory, pattern):
        """Стежити за текою: завершені файли за шаблоном ідуть у R2 самі."""
        if self.enabled:
            self._watches.append((Path(directory), pattern))

    def live(self, path):
        """Файл, що росте весь трен (лог): перезаливати не частіше ``live_every_sec``."""
        if self.enabled:
            self._live[str(path)] = 0.0

    def offer(self, path):
        """Раннер каже: файл дописано, можна вантажити. Не блокує трен."""
        if not self.enabled:
            return
        with self._qlock:
            if str(path) not in self._queue:
                self._queue.append(str(path))
        self._wake.set()

    # ── обхід ───────────────────────────────────────────────────────────────
    def _sig(self, path):
        try:
            st = Path(path).stat()
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def poll_once(self, force=False):
        """Поставити в чергу завершені файли з тек під наглядом і логи, що змінились."""
        if not self.enabled:
            return
        now = self._clock()
        for directory, pattern in self._watches:
            try:
                found = [p for p in directory.glob(pattern) if p.is_file()]
            except OSError:
                continue
            sigs = {p: self._sig(p) for p in found}
            found = sorted((p for p in found if sigs[p] is not None),
                           key=lambda p: (sigs[p][1], p.name))
            for i, p in enumerate(found):
                sig = sigs[p]
                if self._done.get(p.name) == sig:
                    continue
                prev = self._seen.get(str(p))
                self._seen[str(p)] = sig
                newer = i < len(found) - 1
                settled = prev == sig and (now - sig[1] / 1e9) >= self.stable_sec
                if force or newer or settled:
                    self.offer(p)
        for path, last in list(self._live.items()):
            sig = self._sig(path)
            if sig is None or self._done.get(Path(path).name) == sig:
                continue
            if force or now - last >= self.live_every_sec:
                self._live[path] = now
                self.offer(path)

    # ── заливка ─────────────────────────────────────────────────────────────
    def _target(self, name):
        if name in self._named:
            return self._named[name]
        if name in self._slot_of:
            return self._slots[self._slot_of[name]]
        if self._next_slot >= len(self._slots):
            if not self._slots_warned:
                self._slots_warned = True
                self._log(f"[r2-epochs] ⚠ слоти вичерпано ({len(self._slots)}) — "
                          f"{name} і далі лишаються лише на боксі")
            return None
        self._slot_of[name] = self._next_slot
        self._next_slot += 1
        return self._slots[self._slot_of[name]]

    def _put_retry(self, url, body, size, what):
        for attempt in range(self.retries + 1):
            try:
                self._put(url, body, size)
                return True
            except Exception as exc:
                if attempt >= self.retries:
                    self._log(f"[r2-epochs] ⚠ {what}: не залилось після "
                              f"{attempt + 1} спроб ({type(exc).__name__}: {exc}) — трен іде далі")
                    return False
                pause = min(120.0, self.backoff_sec * (2 ** attempt))
                self._log(f"[r2-epochs] {what}: {type(exc).__name__} — повтор за "
                          f"{pause:.0f} с ({attempt + 1}/{self.retries})")
                self._sleep(pause)
        return False

    def _manifest_body(self, final=False):
        return json.dumps({
            "version": 1,
            "job_id": self.spec.get("job_id", ""),
            "final": bool(final),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": self._files,
            "failed": self._failed,
        }, ensure_ascii=False, indent=1).encode("utf-8")

    def _push_manifest(self, final=False):
        m = self.spec.get("manifest") or {}
        if not m.get("url"):
            return False
        body = self._manifest_body(final)
        return self._put_retry(m["url"], body, len(body), "маніфест")

    def upload(self, path):
        """Залити один файл. → True, якщо він тепер у R2 саме в цьому стані."""
        path = Path(path)
        name = path.name
        sig = self._sig(path)
        if sig is None:
            return False
        if self._done.get(name) == sig:
            return True
        target = self._target(name)
        if target is None:
            return False
        try:
            digest = _to_sha256(path)
        except OSError as exc:
            self._log(f"[r2-epochs] ⚠ {name}: не прочитався ({exc})")
            return False
        if self._sig(path) != sig:
            self.offer(path)
            return False
        t0 = self._clock()
        if not self._put_retry(target["url"], str(path), sig[0], name):
            self._failed[name] = target.get("key", "")
            return False
        if self._sig(path) != sig:
            # Файл переписали посеред PUT — у R2 може лежати суміш. Не
            # зараховуємо, стан повториться наступним обходом.
            self._log(f"[r2-epochs] {name} змінився під час заливки — повторю")
            self.offer(path)
            return False
        self._done[name] = sig
        self._failed.pop(name, None)
        self._files[name] = {"key": target.get("key", ""), "size": sig[0], "sha256": digest,
                             "uploaded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        dt = max(0.001, self._clock() - t0)
        self._log(f"[r2-epochs] ✓ {name} ({sig[0] / 2**20:.1f} МБ, {dt:.0f} с) → "
                  f"{target.get('key', '')}")
        self._push_manifest()
        return True

    def drain(self, deadline=None):
        """Залити все, що в черзі. ``deadline`` — абсолютний час, після якого зупинитись."""
        if not self.enabled:
            return
        while True:
            if deadline is not None and self._clock() > deadline:
                return
            if self._stop.is_set() and not self._finished and deadline is None:
                return
            with self._qlock:
                if not self._queue:
                    return
                item = self._queue.pop(0)
            with self._ulock:
                try:
                    self.upload(item)
                except Exception as exc:  # вивантажувач не має права вбити трен
                    self._log(f"[r2-epochs] ⚠ {Path(item).name}: {type(exc).__name__}: {exc}")

    # ── потік ───────────────────────────────────────────────────────────────
    def _loop(self):
        while not self._stop.is_set():
            try:
                self.poll_once()
                self.drain()
            except Exception as exc:
                self._log(f"[r2-epochs] ⚠ обхід: {type(exc).__name__}: {exc}")
            self._wake.wait(self.poll_sec)
            self._wake.clear()

    def start(self):
        if not self.enabled or self._thread is not None:
            return self
        n = len(self._named) + len(self._slots)
        self._log(f"[r2-epochs] увімкнено: {n} посилань → "
                  f"{self.spec.get('prefix', '')} (обхід кожні {self.poll_sec:.0f} с)")
        self._thread = threading.Thread(target=self._loop, name="r2-epochs", daemon=True)
        self._thread.start()
        return self

    def finish(self, globs=(), timeout=1800.0):
        """Кінець трену: усе, що лишилось (і ``globs`` = [(тека, шаблон)]), — у R2.

        Трен уже скінчився, тож стабільності не чекаємо: усе під наглядом
        вважається дописаним.
        """
        if not self.enabled or self._finished:
            return
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
        self._finished = True
        for directory, pattern in globs:
            self.watch(directory, pattern)
        self.poll_once(force=True)
        self.drain(deadline=self._clock() + float(timeout))
        with self._ulock:
            ok = self._push_manifest(final=True)
        with self._qlock:
            left = len(self._queue)
        self._log(f"[r2-epochs] підсумок: у R2 {len(self._files)} файл(ів), "
                  f"не вдалось {len(self._failed)}, не встигло {left}; "
                  f"маніфест {'✓' if ok else '✗'}")


def _to_start(log=None, spec_path=None, **kw):
    """Вивантажувач за специфікацією на боксі; без неї — вимкнений (нічого не робить)."""
    say = log or (lambda m: print(m, flush=True))
    path = Path(spec_path or _TO_SPEC_PATH)
    spec = None
    try:
        if path.is_file():
            spec = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        say(f"[r2-epochs] ⚠ специфікацію {path} не прочитано ({type(exc).__name__}: {exc}) — "
            f"ваги епох лишаються лише на боксі")
        spec = None
    if not spec:
        say("[r2-epochs] вимкнено (специфікації немає) — ваги епох лише у виході")
    return _TrainOutUploader(spec, log=say, **kw)


#: Активний вивантажувач процесу: раннер заводить його там, де знає теки, а
#: `main` закриває в `finally` — і коли трен упав, лог однаково доїде.
_TO_ACTIVE = [None]


def _to_active(log=None, **kw):
    if _TO_ACTIVE[0] is None:
        _TO_ACTIVE[0] = _to_start(log=log, **kw)
    return _TO_ACTIVE[0]


def _to_finish_active(globs=(), timeout=1800.0):
    up = _TO_ACTIVE[0]
    if up is None:
        return
    try:
        up.finish(globs=globs, timeout=timeout)
    except Exception as exc:
        print(f"[r2-epochs] ⚠ фінальне вивантаження: {type(exc).__name__}: {exc}", flush=True)

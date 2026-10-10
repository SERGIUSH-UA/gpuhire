"""Доставка кадрів на орендовану машину: яку частку оренди вона з'їсть.

При транспорті `box` (бакета немає) кадри їдуть з дому прямо на машину, і
весь цей час машина вже тарифікується, але не читає. Година доставки коштує
рівно стільки, скільки година читання, а сторінок не дає жодної.

🔴 Тому межа — частка ОРЕНДИ, а не стелі заходу. Доти доставку міряли проти
`max_hours × 0.25`: на заході зі стелею 8 год машина мала право дві години
лише приймати дані, навіть коли читання займало двадцять хвилин. А за
жорсткої стелі в 2 год межа падала до пів години, і порцію, яку людина вже
погодилась везти, бракувало машина за машиною (10.10.2026, сторонній
користувач на каналі ~1 МБ/с).

Понад межу захід не їде без явної згоди людини (`-p accept_slow_delivery=true`).
Згода — не обхід: людина бачить, скільки годин і доларів піде на перевезення,
і вирішує сама, чи варто воно того.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Яку частку оренди доставка може з'їсти без окремої згоди людини.
DELIVERY_MAX_SHARE = 0.25

#: Ручка плану, якою людина свідомо погоджується платити за перевезення.
ACCEPT_FLAG = "accept_slow_delivery"

#: Замір нижче цієї частки від звичного каналу дому — вада ЦІЄЇ машини
#: (повільний вхідний канал хоста), а не нашого аплінку: таку машину бракуємо й
#: беремо іншу. Ближчий до звичного — це наш канал, і наступна машина отримає
#: те саме, тож захід спиняється.
MACHINE_SIDE_RATIO = 0.5


@dataclass(frozen=True)
class Forecast:
    """Прогноз доставки на одну оренду."""

    nbytes: int
    #: Канал дім → машина, МБ/с; 0 — невідомий.
    mbs: float
    #: Скільки годин машина читатиме цю чергу.
    read_h: float
    #: Ціна години машини; 0 — ще невідома (до вибору).
    dph: float = 0.0
    #: Звідки канал: «замір на цій машині», «попередні доставки».
    source: str = ""

    @property
    def known(self) -> bool:
        return self.mbs > 0 and self.nbytes > 0

    @property
    def hours(self) -> float:
        if not self.known:
            return 0.0
        return self.nbytes / 1e6 / self.mbs / 3600.0

    @property
    def share(self) -> float:
        """Частка оренди, яка піде на доставку."""
        total = self.hours + max(0.0, self.read_h)
        return self.hours / total if total > 0 else 0.0

    @property
    def usd(self) -> float:
        return self.hours * max(0.0, self.dph)

    @property
    def too_costly(self) -> bool:
        return self.known and self.share > DELIVERY_MAX_SHARE

    def fits_bytes(self) -> int:
        """Скільки байтів укладаються в межу за цього каналу й читання."""
        if self.mbs <= 0 or self.read_h <= 0:
            return 0
        hours = self.read_h * DELIVERY_MAX_SHARE / (1.0 - DELIVERY_MAX_SHARE)
        return int(hours * 3600.0 * self.mbs * 1e6)

    def human(self) -> str:
        """Один рядок для людини: обсяг, канал, години, гроші, частка."""
        gb = self.nbytes / 1e9
        if not self.known:
            return (f"доставка {gb:.1f} ГБ на машину: канал ще не міряний — "
                    f"заміряю на першій машині")
        money = f" (~${self.usd:.2f} за ${self.dph:.3f}/год)" if self.dph > 0 else ""
        return (f"доставка {gb:.1f} ГБ при ~{self.mbs:.1f} МБ/с — "
                f"~{hm(self.hours)} оренди без читання{money}; читання "
                f"~{hm(self.read_h)}, тобто на перевезення піде "
                f"{self.share:.0%} оплаченого часу")

    def advice(self) -> str:
        """Що робити людині, коли межу перевищено."""
        fit = self.fits_bytes()
        portion = (f"менші порції — за цього каналу в межу вкладається "
                   f"~{fit / 1e9:.1f} ГБ на захід; " if fit > 0 else "")
        return (f"бакет R2 (`--transport r2`): кадри їдуть у сховище ДО оренди, "
                f"і машина бере їх звідти швидко; або {portion}або свідомо "
                f"заплатити за перевезення: `-p {ACCEPT_FLAG}=true`")

    def as_dict(self) -> dict[str, Any]:
        return {
            "bytes": self.nbytes, "mb_per_sec": round(self.mbs, 2),
            "source": self.source, "hours": round(self.hours, 3),
            "read_hours": round(self.read_h, 3), "share": round(self.share, 3),
            "usd": round(self.usd, 4), "max_share": DELIVERY_MAX_SHARE,
            "too_costly": self.too_costly, "fits_bytes": self.fits_bytes(),
        }


def hm(hours: float) -> str:
    minutes = round(hours * 60)
    return f"{minutes // 60} год {minutes % 60:02d} хв" if minutes >= 60 else f"{minutes} хв"

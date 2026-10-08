#!/usr/bin/env python3
"""Знімок СПРАВЖНЬОГО ринку — щоб алгоритм вибору перевіряли не фікстури.

    python tools/market_snapshot.py                 # знімок у tests/data/market/
    python tools/market_snapshot.py --max-price 0.5 --limit 200

🔴 Навіщо це замість вигаданих офферів. 23.09.2026 наглядач узяв машину за
$0.0270 за ядро-годину там, де тієї ж ночі працювала за $0.0152: на 11%
дешевше за годину й на 78% дорожче за роботу. Жоден синтетичний тест цього не
показав би — у фікстурах пишуть ті оффери, про які вже подумали. Справжній
ринок натомість приносить те, про що не подумали: машини з 96 «ядрами» й
квотою 9.6, картки Pascal за копійки, хости з нульовою надійністю.

Знімок — це просто список офферів, як їх віддає бекенд, плюс дата. Він
КОМІТИТЬСЯ в репозиторій і стає вхідними даними для `tests/test_market_replay.py`:
алгоритм проганяють на ньому без мережі й без оренди.

⚠ Оренди тут немає й бути не може: `search_offers` лише читає каталог.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "data" / "market"

#: Поля, які лишаємо. Решту ріжемо не з таємності, а з ваги: сирий оффер несе
#: десятки полів телеметрії, і знімок на 200 машин розрісся б до мегабайтів у
#: git. Тут — рівно те, що читає `offer_score`.
KEEP = (
    "id", "machine_id", "host_id", "gpu_name", "num_gpus", "gpu_ram",
    "cpu_cores", "cpu_cores_effective", "cpu_ram", "dph_total",
    "disk_space", "inet_down", "inet_up", "reliability2", "geolocation",
    "compute_cap", "rentable", "rented", "duration",
    # Прямі порти й публічна адреса: без них склад на боксі підняти назовні
    # нема куди. Замір 23.09.2026 по 195 офферах — машин без жодного порту
    # НУЛЬ, тож вимога ринок не звужує; але перевіряти її треба ДО оренди.
    "direct_port_count", "public_ipaddr", "static_ip",
)


def trim(offer: dict) -> dict:
    return {k: offer[k] for k in KEEP if k in offer}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.60,
                    help="стеля $/год (ширше за бойову, щоб у знімку були й погані)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--gpu", default="any")
    ap.add_argument("--label", default="", help="суфікс до імені файла")
    args = ap.parse_args()

    from gpurunner.backends.vast import VastBackend

    offers = VastBackend().search_offers(
        gpu=args.gpu, max_price=args.max_price, disk_gb=40,
        num_gpus=1, limit=args.limit,
    )
    day = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    name = f"{day}{('-' + args.label) if args.label else ''}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    payload = {
        "taken_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "source": "vast.ai search_offers",
        "query": {"gpu": args.gpu, "max_price": args.max_price, "limit": args.limit},
        "offers": [trim(o) for o in offers],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding="utf-8")

    cores = [o.get("cpu_cores_effective") or o.get("cpu_cores") or 0 for o in payload["offers"]]
    per_core = sorted(
        (o["dph_total"] / c, o.get("gpu_name"), c)
        for o, c in zip(payload["offers"], cores, strict=False)
        if c and o.get("dph_total")
    )
    print(f"✅ {path}  ·  офферів {len(payload['offers'])}")
    if per_core:
        print(f"   $/ядро-год: від ${per_core[0][0]:.4f} ({per_core[0][1]}, "
              f"{per_core[0][2]:.0f} ядер) до ${per_core[-1][0]:.4f} "
              f"({per_core[-1][1]}, {per_core[-1][2]:.0f} ядер)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Daily implied correlation snapshot: appends one row per tenor to data/implied_corr_history.csv
and keeps the per-name file. Run after the US close.

    python scripts/snapshot_implied_corr.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dispersion.data import load_cboe_index  # noqa: E402
from dispersion.snapshot import implied_snapshot  # noqa: E402
from dispersion.universe import DATA_DIR, load_weights  # noqa: E402

HISTORY = DATA_DIR / "implied_corr_history.csv"


def main() -> None:
    w = load_weights(50, force=True)
    cor = load_cboe_index("COR1M")
    rows = []
    for tenor in (30, 90):
        names, summ = implied_snapshot(w, tenor)
        summ["cboe_cor1m_last"] = float(cor.iloc[-1]) if not cor.empty else None
        summ["earnings_in_tenor"] = int(names["earnings"].notna().sum())
        summ["stale_share"] = float(names["stale"].fillna(False).mean())
        rows.append(summ)
        print(f"{tenor:3d}d  rho_implied={summ['rho_implied']*100:.2f}  index_iv={summ['index_iv']:.1f}  "
              f"avg_single={summ['avg_single_iv']:.1f}  names={summ['names_quoted']}")
    new = pd.DataFrame(rows)
    if HISTORY.exists():
        hist = pd.read_csv(HISTORY)
        merged = pd.concat([hist, new], ignore_index=True).drop_duplicates(subset=["date", "target_days"], keep="last")
    else:
        merged = new
    merged.sort_values(["date", "target_days"]).to_csv(HISTORY, index=False)
    print(f"{len(merged)} rows in {HISTORY}")


if __name__ == "__main__":
    main()

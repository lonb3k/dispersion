"""Today's implied correlation from live option chains, name by name.

For each of the top-50 names and for SPY, take the chain closest to `target_days`, rebuild
implied vols from mids with Black-76 on the put-call-parity forward, read the ATM vol, then
solve the basket identity for rho. Cached to data/implied_snapshot_<date>.csv.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .correlation import ANN, implied_correlation
from .data import load_prices, next_earnings, option_chain
from .options import clean_chain, skew_metrics
from .universe import DATA_DIR


def atm_iv(ticker: str, target_days: int) -> dict:
    raw, expiry, spot = option_chain(ticker, target_days)
    cleaned, info = clean_chain(raw, spot, expiry)
    if cleaned.empty:
        return {"ticker": ticker, "atm_iv": np.nan}
    sk = skew_metrics(cleaned)
    return {"ticker": ticker, "atm_iv": sk["atm_iv"], "rr25": sk["rr25"], "expiry": info["expiry"].date(),
            "days": info["days"], "spot": info["spot"], "stale": bool(info["stale"]), "n_quotes": len(cleaned)}


def implied_snapshot(weights: pd.DataFrame, target_days: int = 30, offline: bool = False,
                     progress=None, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Returns (per-name table, summary). Summary has index_iv, rho_implied, avg_single_iv, dispersion.
    A same-day snapshot file is reused unless `force=True` (chains take about two minutes to read)."""
    today = date.today().isoformat()
    path = DATA_DIR / f"implied_snapshot_{today}_{target_days}d.csv"
    if path.exists() and not force:
        table = pd.read_csv(path)
    else:
        rows = []
        tickers = list(weights["ticker"]) + ["SPY"]
        for i, t in enumerate(tickers):
            try:
                row = atm_iv(t, target_days)
            except Exception as exc:
                row = {"ticker": t, "atm_iv": np.nan, "error": str(exc)[:60]}
            row["earnings"] = next_earnings(t) if t != "SPY" else None
            rows.append(row)
            if progress:
                progress((i + 1) / len(tickers), t)
        table = pd.DataFrame(rows)
        table.to_csv(path, index=False)
    if "earnings" in table:
        table["earnings"] = pd.to_datetime(table["earnings"], errors="coerce")

    # realized 21d vol per name for the IV - RV column
    px = load_prices(list(weights["ticker"]) + ["SPY"], start="2024-01-01", offline=offline)
    r = np.log(px / px.shift(1))
    rv21 = (r.iloc[-21:].std() * np.sqrt(ANN) * 100).rename("rv21")
    table = table.merge(rv21, left_on="ticker", right_index=True, how="left")
    table["iv_minus_rv"] = table["atm_iv"] - table["rv21"]

    names = table[table["ticker"] != "SPY"].merge(weights[["ticker", "name", "sector", "weight"]], on="ticker", how="left")
    ok = names.dropna(subset=["atm_iv"])
    w = ok["weight"] / ok["weight"].sum()
    index_row = table[table["ticker"] == "SPY"].iloc[0]
    index_iv = float(index_row["atm_iv"])
    avg_single = float((w * ok["atm_iv"]).sum())
    rho = implied_correlation(index_iv, ok["atm_iv"].values, w.values)
    rv_index = float(index_row["rv21"]) if pd.notna(index_row["rv21"]) else np.nan
    summary = {"date": today, "index_iv": index_iv, "avg_single_iv": avg_single, "rho_implied": rho,
               "dispersion_iv": avg_single - index_iv, "names_quoted": int(len(ok)), "weight_covered": float(ok["weight"].sum()),
               "index_rv21": rv_index, "target_days": target_days}
    names["vega_weight"] = names["weight"] / names["weight"].sum()
    return names.sort_values("weight", ascending=False).reset_index(drop=True), summary

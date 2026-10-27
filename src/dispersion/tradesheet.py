"""The dispersion book a desk would actually put on today, and what it makes under scenarios.

Vega-weighted variance dispersion: short index variance with vega V, long single-name variances
with vegas w_i V (weights sum to one). In vol points per unit of index vega, the P&L to expiry is

    P&L = sum_i w_i (RV_i - IV_i)  -  (RV_index - IV_index)

Realized index vol is not free: it follows from the single-name realized vols and the realized
correlation through the basket identity. So the whole book reduces to two unknowns, how much
the single names realize relative to their implied (k = RV_i / IV_i) and the realized
correlation rho. The scenario grid below sweeps both. The crash corner (rho up, k up) is where
dispersion books die: correlations go to one and index vol realizes above its implied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .correlation import index_vol_from_correlation

MAG7 = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA"]


def build_book(names: pd.DataFrame, index_iv: float, exclude_earnings: bool = False) -> tuple[pd.DataFrame, dict]:
    """Legs with vega weights. `names` from snapshot.implied_snapshot (atm_iv, rv21, weight, earnings)."""
    legs = names.dropna(subset=["atm_iv"]).copy()
    if exclude_earnings and "earnings" in legs:
        legs = legs[legs["earnings"].isna()]
    legs["vega_weight"] = legs["weight"] / legs["weight"].sum()
    legs["iv_minus_rv"] = legs["atm_iv"] - legs["rv21"]
    legs["carry_contrib_pts"] = legs["vega_weight"] * (legs["rv21"] - legs["atm_iv"])  # long leg P&L if RV = trailing RV
    w, s = legs["vega_weight"].values, legs["atm_iv"].values
    rho_impl = float((index_iv**2 - (w**2 * s**2).sum()) / ((w * s).sum() ** 2 - (w**2 * s**2).sum()))
    avg_single = float((w * s).sum())
    summary = {"index_iv": index_iv, "avg_single_iv": avg_single, "spread_pts": avg_single - index_iv,
               "rho_implied": rho_impl, "legs": int(len(legs)),
               "single_carry_pts": float(legs["carry_contrib_pts"].sum()),
               "mag7_vega_share": float(legs.loc[legs["ticker"].isin(MAG7), "vega_weight"].sum())}
    return legs.sort_values("vega_weight", ascending=False), summary


def scenario_pnl(legs: pd.DataFrame, index_iv: float, rho: float, k: float) -> float:
    """P&L in vol points per unit index vega if every single name realizes k x its implied and
    the realized correlation is rho."""
    w, s = legs["vega_weight"].values, legs["atm_iv"].values
    rv_single = k * s
    rv_index = index_vol_from_correlation(rho, rv_single, w)
    return float((w * (rv_single - s)).sum() - (rv_index - index_iv))


def scenario_grid(legs: pd.DataFrame, index_iv: float,
                  rhos=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80),
                  ks=(0.8, 0.9, 1.0, 1.1, 1.25, 1.5)) -> pd.DataFrame:
    grid = pd.DataFrame(index=[f"rho {r:.0%}" for r in rhos], columns=[f"k {k:.2f}" for k in ks], dtype=float)
    for r in rhos:
        for k in ks:
            grid.loc[f"rho {r:.0%}", f"k {k:.2f}"] = scenario_pnl(legs, index_iv, r, k)
    return grid


def breakeven_correlation(legs: pd.DataFrame, index_iv: float, k: float = 1.0) -> float:
    """Realized correlation at which the book is flat when single names realize k x implied."""
    lo, hi = -0.1, 1.0
    if scenario_pnl(legs, index_iv, lo, k) < 0:
        return float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2
        if scenario_pnl(legs, index_iv, mid, k) > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def expected_pnl_at_trailing(legs: pd.DataFrame, index_iv: float, index_rv: float) -> dict:
    """If every leg realizes exactly its trailing 21d vol."""
    single = float((legs["vega_weight"] * (legs["rv21"] - legs["atm_iv"])).sum())
    index = -(index_rv - index_iv)
    return {"single_legs_pts": single, "index_leg_pts": index, "total_pts": single + index}


def dollar_view(total_pts: float, vega_notional: float) -> float:
    return total_pts * vega_notional

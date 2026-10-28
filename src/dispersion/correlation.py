"""Implied and realized correlation of a weighted basket.

The identity behind every dispersion trade: for a basket with weights w_i and vols s_i and
one average pairwise correlation rho,

    sigma_index^2 = rho * (sum_i w_i s_i)^2 + (1 - rho) * sum_i w_i^2 s_i^2

Solve it for rho and you get the implied correlation when the vols are implied, the realized
correlation when the vols are realized. The CBOE COR indices apply this to the 50 largest S&P
500 names. Index vol is bounded above by the weighted average of single vols (rho = 1): that
gap, "dispersion", is what the trade sells or buys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def implied_correlation(index_vol: float, vols: np.ndarray, weights: np.ndarray) -> float:
    """Average pairwise correlation implied by index vol and single-name vols (same units)."""
    w, s = np.asarray(weights, float), np.asarray(vols, float)
    w = w / w.sum()
    avg = (w * s).sum() ** 2
    diag = (w**2 * s**2).sum()
    return float((index_vol**2 - diag) / (avg - diag))


def index_vol_from_correlation(rho: float, vols: np.ndarray, weights: np.ndarray) -> float:
    w, s = np.asarray(weights, float), np.asarray(vols, float)
    w = w / w.sum()
    return float(np.sqrt(rho * (w * s).sum() ** 2 + (1 - rho) * (w**2 * s**2).sum()))


def weights_frame(weights: pd.Series | pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Daily weights aligned on `prices`: a Series (fixed basket) is broadcast to every date, a DataFrame
    (point-in-time membership) is aligned and forward-filled."""
    if isinstance(weights, pd.DataFrame):
        return weights.reindex(columns=prices.columns).reindex(prices.index).ffill().fillna(0.0)
    w = weights.reindex(prices.columns).fillna(0.0)
    return pd.DataFrame(np.tile(w.values, (len(prices), 1)), index=prices.index, columns=prices.columns)


def realized_correlation_history(prices: pd.DataFrame, weights: pd.Series | pd.DataFrame, window: int = 21,
                                 index_prices: pd.Series | None = None) -> pd.DataFrame:
    """Rolling realized correlation that reconciles index vol with the top-N single-name vols.

    Two conventions, both returned:
    - `rho`: index vol is the realized vol of the *actual index* (SPY when `index_prices` is given).
      This matches the CBOE COR methodology (SPX vol vs top-50 single vols) and is the one to compare
      with COR1M. Because the full index is more diversified than the top-50 basket, this rho is lower.
    - `rho_basket`: index vol is the realized vol of the frozen top-N basket itself. Pure pairwise
      correlation inside the basket, no membership effect.
    Fixed weights (today's), so the basket is a frozen top-N: survivorship is noted in the README.
    Columns: index_vol, basket_vol, avg_single_vol, rho, rho_basket, dispersion (avg_single_vol - index_vol), vol %.
    """
    w0 = weights_frame(weights, prices)
    r = np.log(prices / prices.shift(1))
    var_i = r.rolling(window).var()
    # names without history on a given date (pre-IPO) drop out and weights are renormalized that day
    avail = var_i.notna().astype(float)
    w = avail * w0
    w = w.div(w.sum(axis=1), axis=0)
    basket_r = (r.fillna(0) * w).sum(axis=1)
    var_b = basket_r.rolling(window).var()
    s_i = np.sqrt(var_i)
    avg = (s_i * w).sum(axis=1) ** 2
    diag = (var_i * w**2).sum(axis=1)
    n_names = avail.sum(axis=1)
    if index_prices is not None:
        ri = np.log(index_prices / index_prices.shift(1)).reindex(prices.index)
        var_idx = ri.rolling(window).var()
    else:
        var_idx = var_b
    out = pd.DataFrame({
        "index_vol": np.sqrt(var_idx * ANN) * 100,
        "basket_vol": np.sqrt(var_b * ANN) * 100,
        "avg_single_vol": np.sqrt(avg * ANN) * 100,
        "rho": ((var_idx - diag) / (avg - diag)).clip(-0.2, 1.0),
        "rho_basket": ((var_b - diag) / (avg - diag)).clip(-0.2, 1.0),
        "n_names": n_names,
    }).dropna()
    out["dispersion"] = out["avg_single_vol"] - out["index_vol"]
    return out


def forward_realized_correlation(hist: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Realized correlation over the next `horizon` days, aligned on the start date."""
    return hist["rho"].shift(-horizon)


def pairwise_correlation_matrix(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    r = np.log(prices / prices.shift(1)).iloc[-window:]
    return r.corr()

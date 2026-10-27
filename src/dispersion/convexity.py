"""Selling correlation looks like selling an index put. Measure it.

Correlation spikes exactly when the index falls, so a short correlation swap has the payoff
profile of a short put: steady small gains, large losses in selloffs. We compare the monthly
correlation-swap P&L with the simultaneous S&P return, fit the convexity, and price what the
equivalent put would have paid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def put_sale_pnl(spx: pd.Series, vix: pd.Series, trades_index: pd.DatetimeIndex, horizon: int = 21,
                 delta: float = 0.15) -> pd.DataFrame:
    """Sell a 1-month put struck at `delta` (Black-Scholes at the VIX, no skew) on each trade date,
    hold to expiry. P&L in % of spot."""
    rows = []
    T = horizon / 252
    for d in trades_index:
        if d not in spx.index:
            continue
        i = spx.index.get_loc(d)
        if i + horizon >= len(spx):
            continue
        S0, ST = spx.iloc[i], spx.iloc[i + horizon]
        sigma = vix.reindex([d], method="ffill").iloc[0] / 100
        # strike for a put with |delta| = delta: N(-d1) = delta  ->  d1 = -N^{-1}(delta)
        d1 = -norm.ppf(delta)
        K = S0 * np.exp(-(d1 * sigma * np.sqrt(T) - 0.5 * sigma**2 * T))
        d2 = d1 - sigma * np.sqrt(T)
        premium = K * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        payoff = max(K - ST, 0)
        rows.append({"date": d, "spx_ret_pct": (ST / S0 - 1) * 100, "put_premium_pct": premium / S0 * 100,
                     "put_pnl_pct": (premium - payoff) / S0 * 100, "strike_pct": K / S0 * 100})
    return pd.DataFrame(rows).set_index("date")


def convexity_fit(trades: pd.DataFrame, puts: pd.DataFrame) -> dict:
    """Join correlation-swap P&L (corr points) with SPX return and put P&L; fit pnl = a + b r + c min(r,0)."""
    df = trades[["pnl"]].join(puts, how="inner").dropna()
    df["corr_pnl_pts"] = df["pnl"] * 100
    r = df["spx_ret_pct"].values
    X = np.column_stack([np.ones(len(r)), r, np.minimum(r, 0)])
    beta, *_ = np.linalg.lstsq(X, df["corr_pnl_pts"].values, rcond=None)
    fit = X @ beta
    r2 = 1 - ((df["corr_pnl_pts"] - fit) ** 2).sum() / ((df["corr_pnl_pts"] - df["corr_pnl_pts"].mean()) ** 2).sum()
    corr_with_put = float(np.corrcoef(df["corr_pnl_pts"], df["put_pnl_pct"])[0, 1])
    down = df[df["spx_ret_pct"] < -5]
    return {"data": df, "intercept": float(beta[0]), "slope_up": float(beta[1]), "extra_slope_down": float(beta[2]),
            "r2": float(r2), "corr_with_put": corr_with_put,
            "avg_corr_pnl_all": float(df["corr_pnl_pts"].mean()), "avg_corr_pnl_down5": float(down["corr_pnl_pts"].mean()) if len(down) else np.nan,
            "avg_put_pnl_all": float(df["put_pnl_pct"].mean()), "avg_put_pnl_down5": float(down["put_pnl_pct"].mean()) if len(down) else np.nan,
            "n_down5": int(len(down))}

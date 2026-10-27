"""Correlation risk premium and a correlation-swap backtest on the CBOE COR1M strike.

A dispersion trade (short index variance, long single-name variances, vega-weighted) is in
first order a short position in correlation. The cleanest way to backtest that leg with free
data is a correlation swap: sell correlation at the CBOE implied strike, receive the realized
pairwise correlation of the basket over the following month.

    P&L per month (in correlation points) = K_implied - rho_realized

COR1M exists since 2006, so the test covers 2008, 2011, 2015, 2018, 2020, 2022 and 2024.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .correlation import forward_realized_correlation


def correlation_risk_premium(cor_index: pd.Series, realized: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Daily: implied strike (COR in %), realized correlation over the next `horizon` days, and the premium."""
    impl = cor_index / 100
    fwd = forward_realized_correlation(realized, horizon)
    df = pd.DataFrame({"implied": impl, "realized_fwd": fwd, "realized_trailing": realized["rho"]}).dropna(subset=["implied"])
    df = df[df["realized_trailing"].notna()]
    df["premium"] = df["implied"] - df["realized_fwd"]
    return df


def correlation_swap_backtest(crp: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Non-overlapping trades: one every `horizon` trading days, sell correlation at the implied strike.
    Returns one row per trade with strike, realized, pnl (corr points) and cumulative pnl."""
    trades = crp.dropna(subset=["realized_fwd"]).iloc[::horizon].copy()
    trades["pnl"] = trades["implied"] - trades["realized_fwd"]
    trades["cum_pnl"] = trades["pnl"].cumsum()
    trades["drawdown"] = trades["cum_pnl"] - trades["cum_pnl"].cummax()
    return trades[["implied", "realized_fwd", "pnl", "cum_pnl", "drawdown"]]


def backtest_stats(trades: pd.DataFrame) -> dict:
    p = trades["pnl"]
    monthly_sharpe = p.mean() / p.std() * np.sqrt(12) if p.std() > 0 else np.nan
    worst = trades["pnl"].idxmin()
    return {
        "trades": int(len(p)),
        "avg_pnl_pts": float(p.mean() * 100),
        "hit_ratio": float((p > 0).mean() * 100),
        "sharpe": float(monthly_sharpe),
        "worst_trade_pts": float(p.min() * 100),
        "worst_trade_date": worst,
        "max_drawdown_pts": float(trades["drawdown"].min() * 100),
        "avg_strike": float(trades["implied"].mean() * 100),
        "avg_realized": float(trades["realized_fwd"].mean() * 100),
    }


def premium_by_year(crp: pd.DataFrame) -> pd.DataFrame:
    g = crp.dropna(subset=["premium"]).groupby(crp.dropna(subset=["premium"]).index.year)
    return pd.DataFrame({"avg_implied": g["implied"].mean() * 100, "avg_realized_fwd": g["realized_fwd"].mean() * 100,
                         "avg_premium_pts": g["premium"].mean() * 100, "premium_positive_%": g["premium"].apply(lambda s: (s > 0).mean() * 100)})


def premium_by_vix_regime(crp: pd.DataFrame, vix: pd.Series, edges=(0, 15, 20, 30, 100)) -> pd.DataFrame:
    df = crp.join(vix.rename("vix"), how="inner").dropna(subset=["premium"])
    labels = [f"VIX {a}-{b}" for a, b in zip(edges[:-1], edges[1:])]
    b = pd.cut(df["vix"], edges, labels=labels)
    g = df.groupby(b, observed=True)
    return pd.DataFrame({"days": g.size(), "avg_implied": g["implied"].mean() * 100, "avg_realized_fwd": g["realized_fwd"].mean() * 100,
                         "avg_premium_pts": g["premium"].mean() * 100, "premium_positive_%": g["premium"].apply(lambda s: (s > 0).mean() * 100)})

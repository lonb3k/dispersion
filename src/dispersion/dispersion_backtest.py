"""Historical backtest of the vega-weighted variance dispersion book, rolled monthly.

The book is the one `tradesheet.py` prices on today's chains: short index variance with vega
notional V, long single-name variance on each of the top 50 with vega notional w_i * V (weights sum
to one, so the book is vega-neutral). Each leg is a one-month variance swap:

    N_var = vega / (2 * K)                              variance notional
    payoff = N_var * (RV^2 - K^2)                       long side, RV and K in vol points

RV is the zero-mean realized vol over the 21 trading days of the trade (the variance-swap
convention). Single-name swaps are capped at 2.5x strike, as they trade; the index swap is left
uncapped, which is the conservative side for a seller of index variance.

Strikes. The index strike is VIX (a 30-day variance-swap rate on the S&P 500). Single-name strikes
have no free history: they are rebuilt with `decomposition.implied_moments` (at-the-money level
solved from VIX and COR1M the way CBOE builds COR1M, converted to variance-swap levels with the
smiles measured on the live chains), name by name as k_t x trailing 63-day vol. `strike_check`
compares them with strikes replicated from real chains.

Costs. Each roll pays half a bid-ask on every leg, expressed in vol points of strike: the index is
sold `index_cost` below VIX and every single name is bought `single_cost` above its strike.

P&L is first measured in vol points per unit of index vega notional. To turn it into returns a
capital base is needed: `to_returns` sizes the book with a constant fraction of capital per vega
point, chosen to run at a stated annualized vol. Sharpe does not depend on that choice; CAGR and
drawdown do, and are reported with it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decomposition import C_INDEX_ATM, C_SINGLE_VS, basket_moments, implied_moments

ANN = 252


def _realized_vol_fwd(log_ret: pd.DataFrame | pd.Series, horizon: int) -> pd.DataFrame | pd.Series:
    """Zero-mean realized vol over t+1 .. t+horizon, annualized, vol points, aligned on t."""
    rv2 = (log_ret**2).rolling(horizon).sum() * ANN / horizon * 1e4
    return np.sqrt(rv2).shift(-horizon)


def strikes(prices: pd.DataFrame, weights: pd.Series, index_prices: pd.Series, vix: pd.Series, cor: pd.Series,
            shape_window: int = 63, c_index_atm: float = C_INDEX_ATM, c_single_vs: float = C_SINGLE_VS) -> tuple[pd.Series, pd.DataFrame]:
    """Index strike (VIX) and rebuilt single-name variance-swap strikes, daily."""
    shape = basket_moments(prices, weights, index_prices, shape_window)
    q = implied_moments(vix, cor, shape, c_index_atm, c_single_vs)
    r = np.log(prices / prices.shift(1))
    trail = np.sqrt(r.rolling(shape_window).var() * ANN * 1e4)
    k_single = trail.reindex(q.index).mul(q["k"], axis=0)
    return vix.reindex(q.index), k_single


def backtest(prices: pd.DataFrame, weights: pd.Series, index_prices: pd.Series, vix: pd.Series, cor: pd.Series,
             horizon: int = 21, start: str = "2015-01-01", cap: float = 2.5, index_cost: float = 0.5,
             single_cost: float = 1.0, **calib) -> pd.DataFrame:
    """One row per monthly roll (non-overlapping). P&L in vol points per unit of index vega notional:
    index_leg (short), single_leg (long, vega-weighted), costs, pnl_gross, pnl_net."""
    k_idx, k_sgl = strikes(prices, weights, index_prices, vix, cor, **calib)
    r = np.log(prices / prices.shift(1))
    ri = np.log(index_prices / index_prices.shift(1))
    rv_sgl = _realized_vol_fwd(r, horizon)
    rv_idx = _realized_vol_fwd(ri, horizon)
    dates = k_idx.index[(k_idx.index >= pd.Timestamp(start))]
    dates = [d for d in dates if pd.notna(rv_idx.get(d)) and pd.notna(k_idx.get(d))][::horizon]
    rows = []
    for d in dates:
        K_i, RV_i = k_sgl.loc[d], rv_sgl.loc[d]
        w_d = (weights.loc[:d].iloc[-1] if isinstance(weights, pd.DataFrame) else weights).reindex(K_i.index).fillna(0)
        ok = K_i.notna() & RV_i.notna() & (K_i > 0) & (w_d > 0)
        w = w_d[ok] / w_d[ok].sum()
        K_i, RV_i = K_i[ok], RV_i[ok]
        K_I, RV_I = float(k_idx.loc[d]), float(rv_idx.loc[d])
        # long single-name variance swaps, capped
        rv_capped = np.minimum(RV_i, cap * K_i)
        single_gross = float((w / (2 * K_i) * (rv_capped**2 - K_i**2)).sum())
        single_net = float((w / (2 * (K_i + single_cost)) * (rv_capped**2 - (K_i + single_cost) ** 2)).sum())
        # short index variance swap, uncapped
        index_gross = -1 / (2 * K_I) * (RV_I**2 - K_I**2)
        k_sold = K_I - index_cost
        index_net = -1 / (2 * k_sold) * (RV_I**2 - k_sold**2)
        rows.append({"date": d, "index_strike": K_I, "index_rv": RV_I, "avg_single_strike": float((w * K_i).sum()),
                     "avg_single_rv": float((w * RV_i).sum()), "names": int(ok.sum()),
                     "index_leg": index_gross, "single_leg": single_gross,
                     "pnl_gross": index_gross + single_gross, "pnl_net": index_net + single_net,
                     "capped_names": int((RV_i > cap * K_i).sum())})
    out = pd.DataFrame(rows).set_index("date")
    out["costs"] = out["pnl_gross"] - out["pnl_net"]
    return out


def to_returns(trades: pd.DataFrame, col: str = "pnl_net", target_vol: float = 0.10, sizing: float | None = None) -> tuple[pd.Series, float]:
    """Monthly returns on capital. `sizing` = fraction of capital per vol point of index vega notional
    (0.01: a one-point P&L is 1% of capital). Default: the constant sizing at which the series runs at
    `target_vol` annualized. One scalar for the whole sample: it rescales returns, not their order."""
    pnl = trades[col]
    if sizing is None:
        sizing = target_vol / (pnl.std() * np.sqrt(12))
    return (pnl * sizing).clip(lower=-0.99), float(sizing)


def performance(ret: pd.Series) -> dict:
    """Monthly return series -> CAGR, vol, Sharpe, max drawdown, worst month, hit ratio."""
    cum = (1 + ret).cumprod()
    years = len(ret) / 12
    dd = cum / cum.cummax() - 1
    return {"months": int(len(ret)), "CAGR_%": float((cum.iloc[-1] ** (1 / years) - 1) * 100),
            "ann_vol_%": float(ret.std() * np.sqrt(12) * 100),
            "sharpe": float(ret.mean() / ret.std() * np.sqrt(12)) if ret.std() > 0 else np.nan,
            "max_dd_%": float(dd.min() * 100), "worst_month_%": float(ret.min() * 100),
            "worst_month": ret.idxmin().date(), "hit_%": float((ret > 0).mean() * 100)}


def summary_table(trades: pd.DataFrame, target_vol: float = 0.10) -> pd.DataFrame:
    """Gross and net performance at the same sizing (set on the net series), plus P&L in vol points."""
    ret_net, size = to_returns(trades, "pnl_net", target_vol)
    ret_gross, _ = to_returns(trades, "pnl_gross", sizing=size)
    rows = {"Net of costs": performance(ret_net), "Gross": performance(ret_gross)}
    for k, c in [("Net of costs", "pnl_net"), ("Gross", "pnl_gross")]:
        rows[k]["avg_pnl_vol_pts"] = float(trades[c].mean())
        rows[k]["sizing_%_per_vega_pt"] = size * 100
    return pd.DataFrame(rows).T


def strike_check(k_single: pd.Series, replicated: pd.Series, weights: pd.Series) -> dict:
    """Rebuilt single-name strikes on one date against variance-swap vols replicated from real chains."""
    df = pd.concat({"rebuilt": k_single, "replicated": replicated}, axis=1).join(weights.rename("w")).dropna()
    err = df["rebuilt"] - df["replicated"]
    w = df["w"] / df["w"].sum()
    return {"names": int(len(df)), "weighted_rebuilt": float((w * df["rebuilt"]).sum()),
            "weighted_replicated": float((w * df["replicated"]).sum()),
            "median_abs_error_pts": float(err.abs().median()), "corr_across_names": float(df["rebuilt"].corr(df["replicated"])),
            "table": df.assign(error=err).sort_values("w", ascending=False)}

"""Where does the S&P 500 variance risk premium come from: correlation or single-name vol?

The variance risk premium (VRP) is what the market charges for index variance (VIX squared)
minus the variance the index then realizes over the next 21 trading days. With weights w_i,
single-name vols s_i and one average pairwise correlation rho, index variance is

    V(s, rho) = B + rho * (A - B),   A = (sum_i w_i s_i)^2,   B = sum_i w_i^2 s_i^2

Write V once with what the market prices (Q) and once with what is realized (P). The gap
V(Q) - V(P) splits exactly into a correlation effect and a single-name vol effect. The split is
the Shapley one (average over both orders of changing the two inputs), so it does not depend on
which input is moved first:

    corr_part   = (rho_Q - rho_P) * 0.5 * [(A_Q - B_Q) + (A_P - B_P)]
    single_part = total - corr_part

`corr_part` is what a dispersion trade (short correlation) earns; `single_part` is what a
short position in the 50 single-name variance swaps would earn. A short index variance swap is
the sum of the two.

Implied single-name variances have no free history. They are rebuilt the way CBOE builds COR1M,
then converted to variance-swap levels:
1. COR1M is computed from at-the-money (50 delta) vols. Take an at-the-money proxy for the index,
   c_I * VIX, and solve the identity for the at-the-money level of single-name vols, keeping the
   cross-sectional shape of trailing realized vols (s_ATM,i = k * s_trailing,i).
2. Scale single-name at-the-money vols by c_S to variance-swap levels (the single-name smile).
3. With index variance VIX^2 and these single-name variances, solve the identity once more for the
   model-free implied correlation rho_Q. It sits above COR1M because index skew is steeper than
   single-name skew: crash protection is mostly correlation protection.
c_I and c_S are measured on the days we have the 50 chains (27 Oct 2026: c_I = 0.79 from SPY
model-free variance vs SPY ATM vol, c_S = 1.13 weighted across the 50 names) and treated as
constants; `sensitivity` shows how much the conclusions move with them. Setting both to 1 mixes an
ATM correlation with model-free variances, which overstates single-name implied vols by ~25%.

Units: annualized variance in vol-points squared (VIX 20 -> 400).

Survivorship: the basket is today's top 50, so realized correlation is understated before ~2015.
Results default to 2015+.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def basket_moments(prices: pd.DataFrame, weights: pd.Series, index_prices: pd.Series,
                   window: int = 21) -> pd.DataFrame:
    """Trailing `window`-day realized A, B, index variance and correlation, annualized, vol-pts squared.

    `rho` is unclipped, so `idx_var == B + rho * (A - B)` holds exactly. Names without history on a
    date drop out and weights are renormalized that day (same convention as `realized_correlation_history`).
    """
    w0 = weights.reindex(prices.columns).fillna(0)
    r = np.log(prices / prices.shift(1))
    var_i = r.rolling(window).var() * ANN * 1e4
    avail = var_i.notna().astype(float)
    w = avail.mul(w0, axis=1)
    w = w.div(w.sum(axis=1), axis=0)
    A = (np.sqrt(var_i) * w).sum(axis=1) ** 2
    B = (var_i * w**2).sum(axis=1)
    ri = np.log(index_prices / index_prices.shift(1)).reindex(prices.index)
    idx_var = ri.rolling(window).var() * ANN * 1e4
    out = pd.DataFrame({"A": A, "B": B, "idx_var": idx_var})
    out = out[(avail.sum(axis=1) > 1) & out["idx_var"].notna()]
    out["rho"] = (out["idx_var"] - out["B"]) / (out["A"] - out["B"])
    return out


C_INDEX_ATM = 0.80   # SPY 30d ATM vol / SPY 30d model-free vol (0.79 measured on 27 Oct 2026)
C_SINGLE_VS = 1.10   # single-name 30d model-free vol / ATM vol, weighted (1.13 measured, 1.06 median name)


def implied_moments(vix: pd.Series, cor: pd.Series, shape: pd.DataFrame,
                    c_index_atm: float = C_INDEX_ATM, c_single_vs: float = C_SINGLE_VS) -> pd.DataFrame:
    """Implied (variance-swap) A and B and the model-free implied correlation.

    `vix` and `cor` in points (VIX 20, COR1M 15). `shape` has columns A, B from trailing realized vols:
    only their ratio matters. With c_index_atm = c_single_vs = 1 this is the naive version that
    plugs COR1M straight into VIX.
    """
    df = pd.concat({"vix": vix, "cor": cor}, axis=1).join(shape[["A", "B"]], how="inner").dropna()
    rho_atm = df["cor"] / 100
    var_atm = (df["vix"] * c_index_atm) ** 2
    k2 = var_atm / (df["B"] + rho_atm * (df["A"] - df["B"]))      # ATM level of single-name vols
    A_q, B_q = k2 * df["A"] * c_single_vs**2, k2 * df["B"] * c_single_vs**2
    var_q = df["vix"] ** 2
    rho_q = (var_q - B_q) / (A_q - B_q)                            # model-free implied correlation
    return pd.DataFrame({"var_q": var_q, "rho_q": rho_q, "rho_atm": rho_atm, "A_q": A_q, "B_q": B_q,
                         "k": np.sqrt(k2) * c_single_vs})


def decompose(prices: pd.DataFrame, weights: pd.Series, index_prices: pd.Series, vix: pd.Series, cor: pd.Series,
              horizon: int = 21, shape_window: int = 63, c_index_atm: float = C_INDEX_ATM,
              c_single_vs: float = C_SINGLE_VS) -> pd.DataFrame:
    """Daily decomposition of the index variance premium, aligned on the trade date t.

    Implied side known at t; realized side over the next `horizon` trading days (t+1 .. t+horizon).
    Columns (vol-pts squared unless noted): var_q, var_p, total, corr_part, single_part, rho_q, rho_p,
    A_q, B_q, A_p, B_p, and in vol points: vix, idx_rv_fwd, avg_single_iv, avg_single_rv_fwd.
    """
    shape = basket_moments(prices, weights, index_prices, shape_window)
    fwd = basket_moments(prices, weights, index_prices, horizon).shift(-horizon)
    q = implied_moments(vix, cor, shape, c_index_atm, c_single_vs)
    df = q.join(fwd.rename(columns={"A": "A_p", "B": "B_p", "idx_var": "var_p", "rho": "rho_p"}), how="inner").dropna()
    df["total"] = df["var_q"] - df["var_p"]
    df["corr_part"] = (df["rho_q"] - df["rho_p"]) * 0.5 * ((df["A_q"] - df["B_q"]) + (df["A_p"] - df["B_p"]))
    df["single_part"] = df["total"] - df["corr_part"]
    df["vix"] = np.sqrt(df["var_q"])
    df["idx_rv_fwd"] = np.sqrt(df["var_p"])
    df["avg_single_iv"] = np.sqrt(df["A_q"])
    df["avg_single_rv_fwd"] = np.sqrt(df["A_p"])
    return df


def _since(dec: pd.DataFrame, start: str | None) -> pd.DataFrame:
    return dec if start is None else dec[dec.index >= pd.Timestamp(start)]


def summary(dec: pd.DataFrame, start: str | None = "2015-01-01") -> dict:
    """Average premium and its split, all days since `start`."""
    d = _since(dec, start)
    tot, cp, sp = d["total"].mean(), d["corr_part"].mean(), d["single_part"].mean()
    return {
        "days": int(len(d)), "start": d.index[0], "end": d.index[-1],
        "avg_total": float(tot), "avg_corr": float(cp), "avg_single": float(sp),
        "corr_share_%": float(cp / tot * 100) if tot else np.nan,
        "total_positive_%": float((d["total"] > 0).mean() * 100),
        "corr_positive_%": float((d["corr_part"] > 0).mean() * 100),
        "single_positive_%": float((d["single_part"] > 0).mean() * 100),
        "avg_rho_q_%": float(d["rho_q"].mean() * 100), "avg_rho_p_%": float(d["rho_p"].mean() * 100),
        "avg_single_iv": float(d["avg_single_iv"].mean()), "avg_single_rv_fwd": float(d["avg_single_rv_fwd"].mean()),
        "avg_vix": float(d["vix"].mean()), "avg_idx_rv_fwd": float(d["idx_rv_fwd"].mean()),
    }


def by_period(dec: pd.DataFrame, freq: str = "year", start: str | None = "2015-01-01") -> pd.DataFrame:
    """Average total, correlation and single-name parts by calendar year (or any grouping key series)."""
    d = _since(dec, start)
    key = d.index.year if freq == "year" else freq
    g = d.groupby(key)
    out = pd.DataFrame({"avg_vix": g["vix"].mean(), "avg_rho_q_%": g["rho_q"].mean() * 100, "avg_rho_p_%": g["rho_p"].mean() * 100,
                        "total": g["total"].mean(), "corr_part": g["corr_part"].mean(), "single_part": g["single_part"].mean()})
    out["corr_share_%"] = out["corr_part"] / out["total"] * 100
    return out


def by_vix_regime(dec: pd.DataFrame, edges=(0, 15, 20, 30, 100), start: str | None = "2015-01-01") -> pd.DataFrame:
    d = _since(dec, start)
    labels = [f"VIX {a}-{b}" for a, b in zip(edges[:-1], edges[1:])]
    b = pd.cut(d["vix"], edges, labels=labels)
    g = d.groupby(b, observed=True)
    out = pd.DataFrame({"days": g.size(), "total": g["total"].mean(), "corr_part": g["corr_part"].mean(),
                        "single_part": g["single_part"].mean()})
    out["corr_share_%"] = out["corr_part"] / out["total"] * 100
    return out


def monthly_legs(dec: pd.DataFrame, horizon: int = 21, start: str | None = "2015-01-01") -> pd.DataFrame:
    """Non-overlapping monthly trades: short index variance = short correlation + short single-name variance.
    P&L in variance points per trade, one row per trade, with cumulative sums."""
    d = _since(dec, start).iloc[::horizon][["vix", "rho_q", "rho_p", "total", "corr_part", "single_part"]].copy()
    for c in ["total", "corr_part", "single_part"]:
        d[f"cum_{c}"] = d[c].cumsum()
    return d


def leg_stats(legs: pd.DataFrame) -> pd.DataFrame:
    """Per leg: average, volatility, annualized Sharpe, hit ratio, worst trade and its date, share of the
    5% worst index months explained by that leg."""
    worst = legs["total"] <= legs["total"].quantile(0.05)
    rows = {}
    for c, name in [("total", "Short index variance"), ("corr_part", "Short correlation"), ("single_part", "Short single-name variance")]:
        p = legs[c]
        rows[name] = {"avg": p.mean(), "vol": p.std(), "sharpe": p.mean() / p.std() * np.sqrt(12) if p.std() > 0 else np.nan,
                      "hit_%": (p > 0).mean() * 100, "worst": p.min(), "worst_date": p.idxmin().date(),
                      "share_of_tail_losses_%": p[worst].sum() / legs.loc[worst, "total"].sum() * 100}
    return pd.DataFrame(rows).T


def sensitivity(prices: pd.DataFrame, weights: pd.Series, index_prices: pd.Series, vix: pd.Series, cor: pd.Series,
                starts=("2007-01-01", "2015-01-01", "2024-01-01"),
                grid=((0.75, 1.05), (0.80, 1.10), (0.85, 1.15), (0.80, 1.05), (0.80, 1.15), (1.0, 1.0)),
                shapes=(63,)) -> pd.DataFrame:
    """Correlation share of the premium under alternative (c_I, c_S) calibrations, sample starts and
    shape windows. (1.0, 1.0) is the naive version, kept to show why the conversion matters."""
    rows = []
    for ci, cs in grid:
        for sh in shapes:
            dec = decompose(prices, weights, index_prices, vix, cor, shape_window=sh, c_index_atm=ci, c_single_vs=cs)
            for st in starts:
                s = summary(dec, st)
                rows.append({"c_I": ci, "c_S": cs, "shape window": sh, "from": st[:4],
                             "avg total": s["avg_total"], "avg corr": s["avg_corr"], "avg single": s["avg_single"],
                             "corr share %": s["corr_share_%"]})
    return pd.DataFrame(rows)


# The 31 names of today's top 50 that were already in the S&P 500 top 50 at the start of 2015.
# Using only them removes most of the survivorship bias before 2024 (no NVDA, TSLA, AVGO, AMD... whose
# high pre-megacap vol inflates realized single-name vol), at the cost of a bias after 2024 (the basket
# then misses the names that drive index variance).
INCUMBENTS_2015 = ["AAPL", "MSFT", "XOM", "JNJ", "GE", "AMZN", "WFC", "BRK-B", "JPM", "META", "GOOGL", "GOOG", "PG",
                   "CVX", "KO", "ORCL", "MRK", "INTC", "HD", "BAC", "CSCO", "V", "IBM", "UNH", "AMGN", "MA", "ABBV",
                   "WMT", "PM", "RTX", "LLY"]


def regime_split(prices: pd.DataFrame, weights: pd.Series, index_prices: pd.Series, vix: pd.Series, cor: pd.Series,
                 baskets: dict[str, list[str]], grid=((0.75, 1.05), (0.80, 1.10), (0.85, 1.15)),
                 cut: str = "2024-01-01", start: str = "2015-01-01") -> pd.DataFrame:
    """Average correlation and single-name parts before and after `cut`, for each basket and calibration.

    The question it answers: is the move from correlation to single-name premium after 2024 an artefact
    of one basket or one calibration? Each row is one (basket, c_I, c_S) choice.
    """
    rows = []
    for bname, names in baskets.items():
        names = [t for t in names if t in prices.columns and t in weights.index]
        for ci, cs in grid:
            dec = decompose(prices[names], weights[names], index_prices, vix, cor, c_index_atm=ci, c_single_vs=cs)
            pre = dec[(dec.index >= pd.Timestamp(start)) & (dec.index < pd.Timestamp(cut))]
            post = dec[dec.index >= pd.Timestamp(cut)]
            tail = leg_stats(monthly_legs(pre, start=None)).loc["Short single-name variance", "share_of_tail_losses_%"]
            rows.append({"basket": bname, "c_I": ci, "c_S": cs,
                         f"corr before {cut[:4]}": pre["corr_part"].mean(), f"single before {cut[:4]}": pre["single_part"].mean(),
                         f"corr since {cut[:4]}": post["corr_part"].mean(), f"single since {cut[:4]}": post["single_part"].mean(),
                         "tail losses from single-name vol, before %": tail})
    return pd.DataFrame(rows)

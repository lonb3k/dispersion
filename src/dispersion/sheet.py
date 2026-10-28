"""Morning sheet: the dispersion desk's note, generated from today's data.

`build_sheet()` runs the existing functions (correlation, snapshot, tradesheet, backtest,
convexity, factors) and returns a plain dict: tiles with sparklines, sections of sentences
with the numbers filled in, two to three ideas ranked by historical conditional expectation
against historical worst case, and the existing Plotly figures as JSON for an annex.

Serialized to `sheet/latest.json` by `scripts/build_sheet.py`. No financial logic lives here.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.io as pio

from . import backtest as bt, charts, convexity as cx, factors as fa, tradesheet as ts
from .correlation import realized_correlation_history
from .data import load_cboe_index, load_prices, point_in_time_universe
from .snapshot import implied_snapshot
from .universe import load_weights

REPO_URL = "https://github.com/lonb3k/dispersion"
SPARK_POINTS = 60
CHART_YEARS = 5
MAX_POINTS = 500
BT_START = "2015-01-01"  # survivorship: today's top-50 were not the top-50 before (see README)


# ---------- helpers (same layout as the sister projects) ----------

def _num(x: Any, digits: int = 2) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _spark(s: pd.Series, points: int = SPARK_POINTS) -> list[float]:
    s = s.dropna()
    if s.empty:
        return []
    tail = s.iloc[-points * 2:]
    if len(tail) > points:
        idx = np.linspace(0, len(tail) - 1, points).round().astype(int)
        tail = tail.iloc[idx]
    return [round(float(v), 3) for v in tail.values]


def tile(label: str, value, unit: str = "", sub: str = "", spark: pd.Series | None = None, tone: str = "neutral", digits: int = 1) -> dict:
    return {"label": label, "value": _num(value, digits), "unit": unit, "sub": sub, "tone": tone,
            "spark": _spark(spark) if spark is not None else []}


def _compact(obj):
    if isinstance(obj, list):
        if obj and all(isinstance(v, str) and v.endswith("T00:00:00") for v in obj):
            return [v[:10] for v in obj]
        if obj and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            return [round(v, 3) if isinstance(v, float) else v for v in obj]
        return [_compact(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items()}
    return obj


def _fig(fig, chart_id: str, title: str | None = None) -> dict:
    d = _compact(json.loads(pio.to_json(fig, validate=False)))
    d["layout"].pop("template", None)  # the front end applies its own theme
    return {"id": chart_id, "title": title or fig.layout.title.text, "figure": d}


def _recent(s, years: int = CHART_YEARS, max_points: int = MAX_POINTS):
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    s = s[s.index >= cutoff]
    if len(s) > max_points:
        step = math.ceil(len(s) / max_points)
        s = s.iloc[list(range(len(s) - 1, -1, -step))[::-1]]
    return s


def _pct_rank(s: pd.Series, lookback: int | None = None) -> float:
    w = s.dropna() if lookback is None else s.dropna().iloc[-lookback:]
    return float((w < w.iloc[-1]).mean() * 100) if not w.empty else float("nan")


def _pct_word(p: float) -> str:
    return "very high" if p >= 90 else "high" if p >= 70 else "mid-range" if p >= 30 else "low" if p >= 10 else "very low"


def _score(expected, worst) -> float:
    if expected is None or worst is None or worst >= 0:
        return expected or -1e9
    return expected / abs(worst)


# ---------- the sheet ----------

def build_sheet(offline: bool = False, top_n: int = 50, tenor: int = 30) -> dict:
    today = pd.Timestamp.today().normalize()
    warnings_: list[str] = []
    sections: list[dict] = []
    ideas: list[dict] = []
    chart_list: list[dict] = []

    w = load_weights(top_n)
    tickers = list(w["ticker"]) + ["SPY"]
    px = load_prices(tickers, start="2005-01-01", offline=offline)
    # realized correlation on an approximate point-in-time top 50: a fixed basket of today's names invents a premium
    pit_px, pit_w, pit_spy = point_in_time_universe(offline=offline, top_n=top_n)
    hist = realized_correlation_history(pit_px, pit_w, index_prices=pit_spy)
    cor1m, cor3m, vix = load_cboe_index("COR1M", offline), load_cboe_index("COR3M", offline), load_cboe_index("VIX", offline)
    crp_full = bt.correlation_risk_premium(cor1m, hist)
    crp = crp_full[crp_full.index >= pd.Timestamp(BT_START)]
    trades = bt.correlation_swap_backtest(crp)
    stats = bt.backtest_stats(trades)
    as_of = max(px.index[-1], cor1m.index[-1])

    cor_now, cor3_now = float(cor1m.iloc[-1]), float(cor3m.iloc[-1])
    cor_pct_1y, cor_pct_all = _pct_rank(cor1m, 252), _pct_rank(cor1m)
    rho_real = float(hist["rho"].iloc[-1] * 100)
    rho_real_63 = float(hist["rho"].iloc[-63:].mean() * 100)
    ratio = (cor1m / cor3m).dropna()
    ratio_now = float(ratio.iloc[-1])
    prem_trailing = cor_now - rho_real
    prem_avg = float(crp["premium"].mean() * 100)
    prem_pos = float((crp["premium"] > 0).mean() * 100)
    prem_2y = float(crp[crp.index >= today - pd.DateOffset(years=2)]["premium"].mean() * 100)

    # ---------- today's implied correlation from the chains ----------
    names, summ = implied_snapshot(w, tenor, offline=offline)
    rho_impl = float(summ["rho_implied"] * 100)
    stale_share = float(names["stale"].fillna(False).mean()) if "stale" in names else 0.0
    n_earn = int(names["earnings"].notna().sum()) if "earnings" in names else 0
    legs, book = ts.build_book(names, summ["index_iv"])
    be = ts.breakeven_correlation(legs, summ["index_iv"], 1.0)
    be11 = ts.breakeven_correlation(legs, summ["index_iv"], 1.1)
    carry = ts.expected_pnl_at_trailing(legs, summ["index_iv"], summ["index_rv21"])
    grid = ts.scenario_grid(legs, summ["index_iv"])
    legs_x, book_x = ts.build_book(names, summ["index_iv"], exclude_earnings=True)
    carry_x = ts.expected_pnl_at_trailing(legs_x, summ["index_iv"], summ["index_rv21"]) if book_x["legs"] >= 20 else None

    # ---------- factor views ----------
    def factor_views():
        # same prices as the realized history (one download, one cache), from 2010 like the app
        sub = px.loc[px.index >= "2010-01-01", [t for t in w["ticker"] if t in px]]
        return fa.group_correlations(sub), fa.variance_shares(sub, w.set_index("ticker")["weight"]), fa.first_factor_share(sub)

    try:
        gc, vs, pc = factor_views()
        al = fa.align_with_implied(pc, cor1m)
        al_ = al[al.index >= pd.Timestamp(BT_START)]
    except Exception as exc:
        warnings_.append(f"factors: {type(exc).__name__}: {str(exc)[:120]}")
        gc = vs = pc = al = al_ = None

    # ---------- tiles ----------
    tiles = [
        tile("CBOE COR1M", cor_now, "%", f"{cor_pct_1y:.0f}th pct 1y, {cor_pct_all:.0f}th since 2006", cor1m, "neutral"),
        tile("Implied corr (50 chains)", rho_impl, "%", f"{rho_impl - cor_now:+.1f} vs CBOE, {summ['names_quoted']} names", None, "neutral"),
        tile("Realized corr 21d", rho_real, "%", f"63d avg {rho_real_63:.1f}%", hist["rho"].iloc[-400:] * 100, "neutral"),
        tile("Implied - realized", prem_trailing, " pts", f"avg since 2015 {prem_avg:+.1f}", (cor1m / 100 - hist["rho"]).dropna().iloc[-400:] * 100,
             "pos" if prem_trailing > 0 else "neg"),
        tile("COR1M / COR3M", ratio_now, "", "inverted" if ratio_now > 1 else "normal (3M above 1M)", ratio, "neg" if ratio_now > 1 else "pos", 3),
        tile("SPY ATM IV", summ["index_iv"], "%", f"vs avg single {summ['avg_single_iv']:.1f}%", None, "neutral"),
        tile("Implied dispersion", summ["dispersion_iv"], " pts", "avg single IV - index IV", None, "neutral"),
        tile("Book at trailing vols", carry["total_pts"], " pts", f"singles {carry['single_legs_pts']:+.1f}, index {carry['index_leg_pts']:+.1f}", None,
             "pos" if carry["total_pts"] > 0 else "neg"),
    ]
    if vs is not None:
        tiles.append(tile("Mag 7 share of variance", vs.iloc[-1]["mag7_variance_share"] * 100, "%", f"for {vs.iloc[-1]['mag7_weight']*100:.0f}% of weight",
                          vs["mag7_variance_share"] * 100, "neutral", 0))
    if al is not None:
        tiles.append(tile("First factor share", al.iloc[-1]["factor"] * 100, "%", "variance explained by PC1 (252d)", al["factor"] * 100, "neutral", 0))

    # ---------- sections ----------
    sections.append({
        "title": "Where correlation trades",
        "paragraphs": [
            f"The CBOE one-month implied correlation (COR1M) closed at {cor_now:.1f}%, the {cor_pct_1y:.0f}th percentile of its trailing year and the "
            f"{cor_pct_all:.0f}th since 2006 ({_pct_word(cor_pct_all)}). Solving the basket identity with today's {tenor}-day ATM vols of the "
            f"{summ['names_quoted']} largest S&P 500 names and SPY gives {rho_impl:.1f}%, {abs(rho_impl - cor_now):.1f} points "
            f"{'above' if rho_impl > cor_now else 'below'} the CBOE print.",
            f"Realized correlation of the basket over the last 21 days: {rho_real:.1f}% (63-day average {rho_real_63:.1f}%). Implied sits "
            f"{prem_trailing:+.1f} points {'above' if prem_trailing > 0 else 'below'} trailing realized. Since 2015 the premium over the "
            f"*following* month averaged {prem_avg:+.1f} points and was positive on {prem_pos:.0f}% of days; over the last two years it fell to {prem_2y:+.1f}.",
            f"Term structure: COR3M {cor3_now:.1f}%, ratio COR1M / COR3M {ratio_now:.2f}, "
            + ("inverted: a correlation shock is in progress." if ratio_now > 1 else "normal: correlation is expected to mean-revert up from here."),
        ],
    })
    sections.append({
        "title": "The book today",
        "paragraphs": [
            f"SPY {tenor}-day ATM implied vol {summ['index_iv']:.1f}% against a weighted average single-name vol of {summ['avg_single_iv']:.1f}%: "
            f"{summ['dispersion_iv']:.1f} points of implied dispersion. {n_earn} of {len(names)} names report earnings inside the tenor"
            + (f"; {stale_share*100:.0f}% of the chains had no live bid-ask (read after the close)." if stale_share > 0.5 else "."),
            f"Vega-weighted dispersion book (long {book['legs']} single-name variances, short index variance, Mag 7 = {book['mag7_vega_share']*100:.0f}% of vega): "
            f"breakeven realized correlation {be*100:.1f}% if singles realize their implied, {be11*100:.1f}% if they realize 10% more. "
            f"If every leg realizes its trailing 21-day vol the book makes {carry['total_pts']:+.1f} points per unit of index vega "
            f"(singles {carry['single_legs_pts']:+.1f}, index {carry['index_leg_pts']:+.1f})."
            + (f" Excluding the earnings names ({book_x['legs']} legs) the same read is {carry_x['total_pts']:+.1f} points." if carry_x else ""),
            f"Crash corner of the scenario grid (correlation 80%, singles realizing 1.5x implied): {float(grid.iloc[-1, -1]):+.1f} points. "
            "A crash raises single-name vol and correlation at once, so the index leg loses faster than the single legs gain.",
        ],
        "table": {"columns": ["Name", "Weight", f"ATM IV {tenor}d", "RV 21d", "IV - RV", "Earnings"],
                  "rows": [[r["ticker"], _num(r["weight"] * 100, 1), _num(r["atm_iv"], 1), _num(r["rv21"], 1), _num(r["iv_minus_rv"], 1),
                            r["earnings"].strftime("%Y-%m-%d") if pd.notna(r["earnings"]) else ""] for _, r in names.head(12).iterrows()]},
    })
    if gc is not None:
        last = gc.iloc[-1]
        sections.append({
            "title": "Why correlation is where it is",
            "paragraphs": [
                f"Pairwise 63-day correlation: {last['within_mag7']*100:.0f}% inside the Mag 7, {last['within_rest']*100:.0f}% among the other names, "
                f"{last['cross']*100:.0f}% between the two groups. The Mag 7 carry {vs.iloc[-1]['mag7_variance_share']*100:.0f}% of the basket's variance "
                f"for {vs.iloc[-1]['mag7_weight']*100:.0f}% of its weight.",
                f"The first principal component explains {al.iloc[-1]['factor']*100:.0f}% of the basket's return variance (35-45% in 2020-2022). "
                f"That share and COR1M move together (correlation {al_.corr().iloc[0, 1]:+.2f} since 2015): one strong market factor means expensive "
                "index options and high implied correlation; today's giants trade on their own stories.",
            ],
        })
    # convexity
    puts = cx.put_sale_pnl(px["SPY"], vix, trades.index)
    fit = cx.convexity_fit(trades, puts)
    down = fit["data"][fit["data"]["spx_ret_pct"] < -3]
    sections.append({
        "title": "The short correlation swap, and what it really is",
        "paragraphs": [
            f"Selling a one-month correlation swap at the COR1M strike every 21 trading days since 2015: {stats['trades']} trades, "
            f"{stats['avg_pnl_pts']:+.1f} points per trade on average, {stats['hit_ratio']:.0f}% winners, worst {stats['worst_trade_pts']:+.1f} points in "
            f"{stats['worst_trade_date'].strftime('%B %Y')}, max drawdown {stats['max_drawdown_pts']:+.1f} points. Gross of a bid-ask that runs several points.",
            f"Regressed on the S&P return of the same month, the swap P&L moves {fit['slope_up']:+.2f} points per 1% of index move and "
            f"{fit['extra_slope_down']:+.2f} more per 1% on the way down (R² {fit['r2']:.2f}); its correlation with a short 15-delta put sold at the VIX is "
            f"{fit['corr_with_put']:+.2f}. In the {len(down)} months where the index fell more than 3% the swap lost {down['corr_pnl_pts'].mean():+.1f} points "
            f"against {fit['avg_corr_pnl_all']:+.1f} overall. Selling correlation is selling crash insurance with a different label.",
        ],
    })

    # ---------- ideas ----------
    # bucket history by the level of the strike (quintiles of COR1M since 2006) and read the forward premium in today's bucket
    edges = list(cor1m.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values)
    edges[0], edges[-1] = -1, 200
    labels = ["bottom quintile", "2nd quintile", "3rd quintile", "4th quintile", "top quintile"]
    crp_b = crp_full.copy()
    crp_b["bucket"] = pd.cut(crp_b["implied"] * 100, edges, labels=labels)
    bucket_now = str(pd.cut([cor_now], edges, labels=labels)[0])
    in_b = crp_b[(crp_b["bucket"].astype(str) == bucket_now)].dropna(subset=["premium"])
    tr_b = trades.join(crp_b["bucket"], how="left")
    tr_in = tr_b[tr_b["bucket"].astype(str) == bucket_now]
    if len(in_b) >= 60:
        exp_b = float(in_b["premium"].mean() * 100)
        worst_tr = float(tr_in["pnl"].min() * 100) if len(tr_in) else float(in_b["premium"].min() * 100)
        worst_date = tr_in["pnl"].idxmin() if len(tr_in) else in_b["premium"].idxmin()
        hit = float((tr_in["pnl"] > 0).mean() * 100) if len(tr_in) else float((in_b["premium"] > 0).mean() * 100)
        ideas.append({
            "title": "Sell the 1-month correlation swap at the COR1M strike",
            "direction": "short correlation", "instrument": f"1m correlation swap on the top-{top_n}, strike {cor_now:.1f}%",
            "condition": f"COR1M in its {bucket_now} since 2006 ({cor_pct_all:.0f}th percentile)",
            "thesis": f"When the strike sat in this quintile, implied exceeded the realized correlation of the following month by {exp_b:+.1f} points on "
                      f"average ({float((in_b['premium'] > 0).mean() * 100):.0f}% of days). Today's strike is {prem_trailing:+.1f} points above trailing realized.",
            "stats": {"expected": _num(exp_b, 2), "expected_unit": "corr pts per month", "hit": _num(hit, 0), "worst": _num(worst_tr, 1),
                      "worst_unit": "corr pts (one trade)", "worst_date": worst_date.strftime("%b %Y"), "n": int(len(tr_in)) or int(len(in_b)),
                      "sample": f"{len(tr_in)} monthly trades and {len(in_b)} daily observations in this bucket since 2006",
                      "unconditional": f"all {stats['trades']} trades since 2015: {stats['avg_pnl_pts']:+.1f} pts avg, {stats['hit_ratio']:.0f}% hit, worst {stats['worst_trade_pts']:+.1f} ({stats['worst_trade_date'].strftime('%b %Y')})"},
        })
        # the mirror trade: buy correlation when it is this cheap
        ideas.append({
            "title": "Buy correlation (short dispersion) as a crash hedge",
            "direction": "long correlation", "instrument": "receive the 1m correlation swap, or short singles / long index variance",
            "condition": f"COR1M at {cor_now:.1f}%, {bucket_now}; COR1M / COR3M {ratio_now:.2f}",
            "thesis": f"The hedge costs the premium: {-exp_b:+.1f} points per month on average in this bucket. It pays when correlation jumps; the best "
                      f"month for the buyer in this bucket returned {-worst_tr:+.1f} points ({worst_date.strftime('%b %Y')}). "
                      "Cheap insurance by history, negative carry by construction.",
            "stats": {"expected": _num(-exp_b, 2), "expected_unit": "corr pts per month", "hit": _num(100 - hit, 0),
                      "worst": _num(-float(tr_in["pnl"].max() * 100), 1) if len(tr_in) else None, "worst_unit": "corr pts (one trade)",
                      "worst_date": tr_in["pnl"].idxmax().strftime("%b %Y") if len(tr_in) else None, "n": int(len(tr_in)),
                      "sample": f"{len(tr_in)} monthly trades in this bucket since 2006"},
        })
    # the same book rolled monthly since 2015 on rebuilt strikes and the point-in-time universe
    book_hist = None
    try:
        from . import dispersion_backtest as dbt
        bk = dbt.backtest(pit_px, pit_w, pit_spy, vix, cor1m, start=BT_START)
        bk_net = dbt.performance(dbt.to_returns(bk, "pnl_net")[0])
        bk_gross = dbt.performance(dbt.to_returns(bk, "pnl_gross")[0])
        book_hist = (f"rolled monthly since 2015 on rebuilt strikes: {bk['pnl_gross'].mean():+.1f} vol pts gross (Sharpe {bk_gross['sharpe']:.2f}), "
                     f"{bk['pnl_net'].mean():+.1f} after 0.5 / 1 vol pt of costs (Sharpe {bk_net['sharpe']:.2f})")
    except Exception as exc:
        warnings_.append(f"book backtest: {type(exc).__name__}: {str(exc)[:120]}")
    # the vega-weighted book read at trailing vols, worst case = crash corner of the grid (a scenario, labelled as such)
    ideas.append({
        "title": "Put on the vega-weighted dispersion book" + (" excluding earnings names" if carry_x and carry_x["total_pts"] > carry["total_pts"] else ""),
        "direction": "long single-name variance, short index variance",
        "instrument": f"{(book_x if carry_x and carry_x['total_pts'] > carry['total_pts'] else book)['legs']} legs, vega-neutral, {tenor}-day tenor",
        "condition": f"Implied dispersion {summ['dispersion_iv']:.1f} pts; {n_earn} names in earnings; breakeven correlation {be*100:.1f}%",
        "thesis": f"At trailing realized vols the book makes {(carry_x if carry_x and carry_x['total_pts'] > carry['total_pts'] else carry)['total_pts']:+.1f} points per unit of index vega. "
                  + ("Single names are realizing below their implied because earnings are priced but not printed: the trade pays only if the prints move the stocks."
                     if carry["single_legs_pts"] < 0 else "Single names are realizing above their implied: the long legs carry positively."),
        "stats": {"expected": _num((carry_x if carry_x and carry_x["total_pts"] > carry["total_pts"] else carry)["total_pts"], 2),
                  "expected_unit": "vol pts per unit index vega (at trailing vols)", "hit": None,
                  "worst": _num(float(grid.iloc[-1, -1]), 1), "worst_unit": "vol pts, crash scenario (rho 80%, k 1.5)", "worst_date": "scenario",
                  "n": int(book["legs"]), "sample": "today's chains; the worst case is a scenario, not a historical observation",
                  **({"unconditional": book_hist} if book_hist else {})},
    })
    for i in ideas:
        i["score"] = _num(_score(i["stats"].get("expected"), i["stats"].get("worst")), 3)
    ideas.sort(key=lambda d: d["score"] if d["score"] is not None else -1e9, reverse=True)
    for k, i in enumerate(ideas, 1):
        i["rank"] = k

    # ---------- charts ----------
    chart_list.append(_fig(charts.implied_vs_realized(_recent(crp)), "implied_vs_realized"))
    chart_list.append(_fig(charts.premium_area(_recent(crp)), "premium"))
    chart_list.append(_fig(charts.names_iv_rv(names), "names_iv_rv"))
    chart_list.append(_fig(charts.scenario_heatmap(grid), "scenario_grid"))
    chart_list.append(_fig(charts.cumulative_pnl(trades), "swap_cum_pnl"))
    chart_list.append(_fig(charts.convexity_scatter(fit), "convexity"))
    if al_ is not None:
        chart_list.append(_fig(charts.factor_vs_implied(_recent(al_)), "factor_vs_implied"))
        chart_list.append(_fig(charts.group_corr_lines(_recent(gc[gc.index >= pd.Timestamp(BT_START)])), "group_corr"))
    chart_list.append(_fig(charts.corr_term_structure(_recent(cor1m), _recent(cor3m)), "corr_term_structure"))

    lede = (f"{as_of.strftime('%A %d %B %Y')}. COR1M {cor_now:.1f}% ({_pct_word(cor_pct_all)} since 2006), our read from {summ['names_quoted']} chains "
            f"{rho_impl:.1f}%, realized {rho_real:.1f}%. Implied dispersion {summ['dispersion_iv']:.1f} points with {n_earn} names in earnings; "
            f"the book at trailing vols is {carry['total_pts']:+.1f} points.")
    if ideas:
        lede += f" Top idea: {ideas[0]['title'][0].lower() + ideas[0]['title'][1:]}."

    return {
        "project": "dispersion", "slug": "dispersion", "title": "Dispersion", "subtitle": "Implied vs realized correlation on the S&P 500 top-50",
        "as_of": as_of.strftime("%Y-%m-%d"), "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "headline": lede, "tiles": tiles, "sections": sections, "ideas": ideas, "charts": chart_list,
        "method": "Ideas are ranked by historical conditional expectation divided by the historical worst case in the same regime. "
                  "Conditioning variable: the quintile of the COR1M strike since 2006. The dispersion book's worst case is the crash corner of the "
                  "scenario grid, labelled as a scenario.",
        "caveats": [
            "Realized correlation uses an approximate point-in-time top 50 (today's float shares times past prices, rebalanced monthly). A fixed "
            "basket of today's names would understate past correlation and show a premium of about 13 points per month that is not there.",
            "Correlation swap P&L is gross: no bid-ask (several points on a real swap), no margin, frozen basket.",
            "Option quotes from Yahoo are delayed; after the close most chains have no bid-ask and IVs come from last trades.",
        ],
        "warnings": warnings_,
        "links": {"repo": REPO_URL, "streamlit": "https://lonb3k-dispersion.streamlit.app"},
    }

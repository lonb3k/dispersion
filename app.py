"""Equity dispersion: implied vs realized correlation on the S&P 500.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dispersion import backtest as bt, charts, convexity as cx, decomposition as dcmp, factors as fa, tradesheet as ts  # noqa: E402
from dispersion.correlation import pairwise_correlation_matrix, realized_correlation_history  # noqa: E402
from dispersion.data import load_cboe_index, load_prices, point_in_time_universe  # noqa: E402
from dispersion import dispersion_backtest as dbt  # noqa: E402
from dispersion.snapshot import implied_snapshot  # noqa: E402
from dispersion.universe import DATA_DIR, load_weights  # noqa: E402

st.set_page_config(page_title="Dispersion", page_icon="🔀", layout="wide")

# Shared hosts get rate-limited by Yahoo, so there the app opens on the CSV mirrors committed to the repo.
# Streamlit Community Cloud mounts the repo under /mount/src, Hugging Face Spaces sets SPACE_ID.
HOSTED = (str(Path(__file__).resolve()).startswith("/mount/src/") or "SPACE_ID" in os.environ
          or os.environ.get("OFFLINE_DEFAULT") == "1")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def weights(top_n: int) -> pd.DataFrame:
    return load_weights(top_n)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def prices(tickers: tuple[str, ...], start: str, offline: bool) -> pd.DataFrame:
    return load_prices(list(tickers), start=start, offline=offline)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cboe(index: str, offline: bool) -> pd.Series:
    return load_cboe_index(index, offline=offline)


@st.cache_data(ttl=3600, show_spinner=False)
def snapshot(top_n: int, target_days: int, offline: bool):
    w = load_weights(top_n)
    return implied_snapshot(w, target_days, offline=offline)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def factor_views(tickers: tuple[str, ...], start: str, offline: bool):
    w = load_weights(len(tickers))
    px = load_prices(list(tickers) + ["SPY"], start=start, offline=offline)
    sub = px[[t for t in tickers if t in px]]
    return (fa.group_correlations(sub), fa.variance_shares(sub, w.set_index("ticker")["weight"]), fa.first_factor_share(sub))


@st.cache_data(ttl=6 * 3600, show_spinner="Rebuilding the point-in-time top 50...")
def pit_universe(offline: bool):
    return point_in_time_universe(offline=offline)


@st.cache_data(ttl=6 * 3600, show_spinner="Decomposing 20 years of variance premium...")
def premium_decomposition(offline: bool):
    """Point-in-time top 50 as the main case; today's top 50 and the 2015 incumbents as fixed baskets, to show survivorship."""
    P, W, spy = point_in_time_universe(offline=offline)
    w50 = load_weights(50).set_index("ticker")["weight"]
    names = [t for t in w50.index if t in P]
    inc = [t for t in dcmp.INCUMBENTS_2015 if t in P]
    v, c = load_cboe_index("VIX", offline=offline), load_cboe_index("COR1M", offline=offline)
    dec = dcmp.decompose(P, W, spy, v, c)
    rob = dcmp.regime_split(spy, v, c, {"Point-in-time top 50": (P, W), "Today's top 50 (survivors)": (P[names], w50[names]),
                                        "31 incumbents of 2015": (P[inc], w50[inc])})
    return dec, rob


@st.cache_data(ttl=6 * 3600, show_spinner="Rolling the dispersion book since 2015...")
def dispersion_book(offline: bool):
    P, W, spy = point_in_time_universe(offline=offline)
    v, c = load_cboe_index("VIX", offline=offline), load_cboe_index("COR1M", offline=offline)
    base = dbt.backtest(P, W, spy, v, c)
    rows = []
    for cs in (1.05, 1.10, 1.15):
        for ic_, sc in [(0.0, 0.0), (0.5, 1.0), (0.5, 2.0)]:
            t = dbt.backtest(P, W, spy, v, c, index_cost=ic_, single_cost=sc, c_single_vs=cs)
            p = dbt.performance(dbt.to_returns(t, "pnl_net")[0])
            rows.append({"c_S (single smile)": cs, "costs, index / single (vol pts)": f"{ic_} / {sc}",
                         "avg P&L (vol pts)": t["pnl_net"].mean(), "Sharpe": p["sharpe"], "hit %": p["hit_%"],
                         "worst month at 10% vol, %": p["worst_month_%"], "max drawdown at 10% vol, %": p["max_dd_%"]})
    w50 = load_weights(50).set_index("ticker")["weight"]
    names = [t for t in w50.index if t in P]
    survivors = dbt.backtest(P[names], w50[names], spy, v, c)
    return base, pd.DataFrame(rows), survivors


def fmt(x, suffix="%", d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}{suffix}"


with st.sidebar:
    st.title("Dispersion")
    st.caption("Implied vs realized correlation on the S&P 500 top-50, the CBOE COR1M way.")
    top_n = st.slider("Basket size (largest by weight)", 20, 50, 50, step=5)
    offline = st.toggle("Offline (cached data only)", value=HOSTED,
                        help="Off: live Yahoo prices and option chains (about two minutes for 51 chains). On: the CSV extracts in data/.")
    if HOSTED and offline:
        st.caption("Hosted demo: opens on the CSV snapshot committed to the repo (implied vols from the last daily run). "
                   "Switch offline off for live data.")
    bt_start = st.date_input("Backtest from", pd.Timestamp("2015-01-01"))
    basket_mode = st.radio("Basket for realized correlation", ["Point-in-time top 50", "Today's top 50 (survivorship bias)"], 0,
                           help="Point in time: the 50 largest names of each month, rebuilt from today's float shares and past prices. "
                                "Today's top 50: the naive fixed basket, kept to show how much survivorship inflates the premium.")
    st.divider()
    st.caption("Prices: Yahoo Finance. Weights: SPY holdings (State Street). Implied correlation history: CBOE COR1M / COR3M. "
               "Single-name implied vols rebuilt from option mids with Black-76 on the put-call-parity forward.")

w = weights(top_n)
tickers = tuple(w["ticker"]) + ("SPY",)
px = prices(tickers, "2005-01-01", offline)
if basket_mode.startswith("Point"):
    _P, _W, _spy = pit_universe(offline)
    hist = realized_correlation_history(_P, _W, index_prices=_spy)
else:
    hist = realized_correlation_history(px[list(w["ticker"])], w.set_index("ticker")["weight"], index_prices=px["SPY"])
cor1m, cor3m, vix = cboe("COR1M", offline), cboe("COR3M", offline), cboe("VIX", offline)
crp_full = bt.correlation_risk_premium(cor1m, hist)

st.title("Equity dispersion: is correlation too expensive?")
st.caption(f"Basket: top {top_n} S&P 500 names covering {w['index_weight'].sum()*100:.0f}% of the index, re-weighted to 100%. "
           f"Last price {px.index[-1].date()}, last COR1M {cor1m.index[-1].date()}.")

tabs = st.tabs(["Today", "Trade sheet", "Why so low", "Implied vs realized", "Correlation swap backtest",
                "Where the premium comes from", "Mechanics"])

# ---------- Today ----------
with tabs[0]:
    tenors = [30, 60, 90]
    if offline:  # only the tenors with a committed snapshot
        tenors = [d for d in tenors if any(DATA_DIR.glob(f"implied_snapshot_*_{d}d.csv"))] or tenors
    target = st.select_slider("Option tenor (days)", tenors, value=30) if len(tenors) > 1 else tenors[0]
    with st.spinner("Reading option chains name by name (about two minutes the first time)..."):
        names, summ = snapshot(top_n, target, offline)
    if offline:
        st.caption(f"Option chains read on {summ['date']} (latest committed snapshot).")
    last_cor = float(cor1m.iloc[-1])
    c = st.columns(5)
    c[0].metric("Implied correlation (ours)", fmt(summ["rho_implied"] * 100))
    c[1].metric("CBOE COR1M (last close)", fmt(last_cor), f"{summ['rho_implied']*100 - last_cor:+.1f} gap", delta_color="off")
    c[2].metric("SPY ATM IV", fmt(summ["index_iv"]))
    c[3].metric("Avg single-name IV", fmt(summ["avg_single_iv"]), "weighted", delta_color="off")
    c[4].metric("Implied dispersion", fmt(summ["dispersion_iv"], " pts"), "avg single - index", delta_color="off")
    realized_now = float(hist["rho"].iloc[-1] * 100)
    st.markdown(
        f"Solving the basket identity with today's ATM vols gives **{summ['rho_implied']*100:.1f}%** implied correlation, "
        f"against **{last_cor:.1f}%** for the CBOE print and **{realized_now:.1f}%** realized over the last 21 days. "
        f"{int(names['earnings'].notna().sum())} of {len(names)} names report earnings within the tenor: single-name vol is "
        "bid into results, which mechanically pushes implied correlation down. Dispersion is seasonally cheap to buy and expensive to sell in earnings season."
    )
    if names["stale"].fillna(False).mean() > 0.5:
        st.warning("Market closed: most quotes come from last trades, not live bid/ask. Rerun during US hours for a clean read.")
    st.plotly_chart(charts.names_iv_rv(names), use_container_width=True)
    st.caption("Violet bars: earnings inside the tenor. Green: single-name vol rich vs its own realized. Red: cheap. "
               "A dispersion desk buys the rich-looking names' vol only if the premium survives the earnings move.")
    with st.expander("Per-name table"):
        show = names[["ticker", "name", "sector", "weight", "atm_iv", "rv21", "iv_minus_rv", "rr25", "earnings", "n_quotes", "stale"]].copy()
        st.dataframe(show.round(3), use_container_width=True, hide_index=True,
                     column_config={"weight": st.column_config.NumberColumn(format="%.2%")})


# ---------- Trade sheet ----------
with tabs[1]:
    st.subheader("The dispersion book, vega-weighted, with today's implied vols")
    c1, c2, c3 = st.columns([1, 1, 1])
    excl = c1.toggle("Exclude names with earnings inside the tenor", value=False)
    vega_notional = c2.number_input("Index vega notional ($ per vol point)", 10_000, 10_000_000, 100_000, step=10_000)
    legs, book = ts.build_book(names, summ["index_iv"], exclude_earnings=excl)
    if book["legs"] < 20:
        st.warning(f"Only {book['legs']} legs left: the implied correlation of a basket that small is not meaningful. Untoggle to see the full book.")
    be = ts.breakeven_correlation(legs, summ["index_iv"], 1.0)
    be11 = ts.breakeven_correlation(legs, summ["index_iv"], 1.1)
    carry = ts.expected_pnl_at_trailing(legs, summ["index_iv"], summ["index_rv21"])
    c = st.columns(5)
    c[0].metric("Legs", book["legs"], f"Mag 7 = {book['mag7_vega_share']*100:.0f}% of vega", delta_color="off")
    c[1].metric("Implied spread", fmt(book["spread_pts"], " pts"), "avg single IV − index IV", delta_color="off")
    c[2].metric("Breakeven correlation", fmt(be * 100), f"{be11*100:.1f}% if singles realize +10%", delta_color="off")
    c[3].metric("P&L at trailing vols", fmt(carry["total_pts"], " pts"), f"singles {carry['single_legs_pts']:+.1f}, index {carry['index_leg_pts']:+.1f}", delta_color="off")
    c[4].metric("Same in dollars", f"${ts.dollar_view(carry['total_pts'], vega_notional):,.0f}", f"on ${vega_notional:,.0f} vega", delta_color="off")
    st.plotly_chart(charts.scenario_heatmap(ts.scenario_grid(legs, summ["index_iv"])), use_container_width=True)
    st.markdown(
        f"**Reading the grid.** Long single-name variance, short index variance, vega-neutral. The book is flat on the diagonal where realized "
        f"correlation equals the implied {book['rho_implied']*100:.1f}% and singles realize their implied (k = 1). It makes money when singles move "
        "more than priced (earnings, idiosyncratic news) or when correlation stays low; it loses when correlation jumps, and the loss accelerates "
        "with k because a crash raises single-name vol *and* correlation at once, so the index leg loses faster than the single legs gain. "
        f"At today's trailing realized vols the book would make **{carry['total_pts']:+.1f} pts**: single names are realizing "
        f"{-carry['single_legs_pts']:.1f} pts below their implied because earnings are priced but not yet printed. The trade only pays if the prints move the stocks."
    )
    st.plotly_chart(charts.legs_bar(legs), use_container_width=True)
    with st.expander("Legs"):
        st.dataframe(legs[["ticker", "name", "sector", "vega_weight", "atm_iv", "rv21", "iv_minus_rv", "rr25", "earnings"]].round(3),
                     use_container_width=True, hide_index=True, column_config={"vega_weight": st.column_config.NumberColumn(format="%.2%")})

# ---------- Why so low ----------
with tabs[2]:
    st.subheader("Why implied correlation sits at record lows: the index became a few uncorrelated giants")
    with st.spinner("Rolling correlations and principal components..."):
        gc, vs, pc = factor_views(tuple(w["ticker"]), "2010-01-01", offline)
    gc_ = gc[gc.index >= pd.Timestamp(bt_start)]
    vs_ = vs[vs.index >= pd.Timestamp(bt_start)]
    last = gc.iloc[-1]
    c = st.columns(4)
    c[0].metric("Within Mag 7", fmt(last["within_mag7"] * 100), "63d pairwise", delta_color="off")
    c[1].metric("Within the other names", fmt(last["within_rest"] * 100))
    c[2].metric("Mag 7 vs the rest", fmt(last["cross"] * 100))
    c[3].metric("Mag 7 share of basket variance", fmt(vs.iloc[-1]["mag7_variance_share"] * 100), f"for {vs.iloc[-1]['mag7_weight']*100:.0f}% of weight", delta_color="off")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.group_corr_lines(gc_), use_container_width=True)
    c2.plotly_chart(charts.variance_share_lines(vs_), use_container_width=True)
    al = fa.align_with_implied(pc, cor1m)
    al_ = al[al.index >= pd.Timestamp(bt_start)]
    st.plotly_chart(charts.factor_vs_implied(al_), use_container_width=True)
    st.markdown(
        f"The first principal component explains **{al.iloc[-1]['factor']*100:.0f}%** of the basket's return variance today, against 35-45% in "
        f"2020-2022. Correlation between that share and COR1M since {al_.index[0].year}: **{al_.corr().iloc[0,1]:+.2f}**. "
        "When one factor drives the market, index options are expensive relative to single names and implied correlation is high. "
        "Today the giants trade on their own stories (AI capex, autos, ads), the market factor is weak, and the index is cheap insurance. "
        "That is the structural reason the dispersion premium collapsed in 2024-2025, and the reason a reversal (one macro shock that "
        "re-couples the Mag 7) is the trade's real risk."
    )

# ---------- Implied vs realized ----------
with tabs[3]:
    crp = crp_full[crp_full.index >= pd.Timestamp(bt_start)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg implied (COR1M)", fmt(crp["implied"].mean() * 100))
    c2.metric("Avg realized next 21d", fmt(crp["realized_fwd"].mean() * 100))
    c3.metric("Avg premium", fmt(crp["premium"].mean() * 100, " pts"))
    c4.metric("Premium > 0", fmt((crp["premium"] > 0).mean() * 100, "%", 0), "of days", delta_color="off")
    st.plotly_chart(charts.implied_vs_realized(crp), use_container_width=True)
    st.plotly_chart(charts.premium_area(crp), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**By year**")
        st.dataframe(bt.premium_by_year(crp).round(1), use_container_width=True)
    with c2:
        st.markdown("**By VIX regime**")
        st.dataframe(bt.premium_by_vix_regime(crp, vix).round(1), use_container_width=True)
        if basket_mode.startswith("Point"):
            st.markdown(
                "On a point-in-time basket, at-the-money implied correlation has been an almost unbiased forecast of next month's "
                "realized correlation since 2015: the premium averages a fraction of a point and is positive on about half the days. "
                "It only appears in stress (VIX above 30). The index variance premium is real, but it does not sit in at-the-money "
                "correlation: see *Where the premium comes from*."
            )
        else:
            st.markdown(
                "This is the survivorship trap. Today's top 50 include names that were small, volatile and loosely correlated ten "
                "years ago (NVDA, TSLA, AVGO, PLTR...), so their past realized correlation is far below that of the index's actual "
                "top 50, and implied correlation looks 10 to 15 points too expensive. Switch the sidebar back to point in time."
            )
    st.info(
        "**Point-in-time basket.** Each month, the 50 largest names among today's 250 largest, by today's float-adjusted share "
        "count times that month's split-adjusted price, as CBOE does for COR1M. Known gaps: share counts move (heavy "
        "issuers such as Citi in 2008-2009 look too big in the past, heavy buyers such as Apple too small), and names delisted "
        "since (Celgene, EMC...) cannot enter. In 2026 the two baskets share 47 names and realize the same correlation; in 2015 "
        "they share 29 and differ by 20 points."
    )

# ---------- Backtest ----------
with tabs[4]:
    crp = crp_full[crp_full.index >= pd.Timestamp(bt_start)]
    trades = bt.correlation_swap_backtest(crp)
    stats = bt.backtest_stats(trades)
    c = st.columns(6)
    c[0].metric("Trades (monthly)", stats["trades"])
    c[1].metric("Avg P&L / trade", fmt(stats["avg_pnl_pts"], " pts"))
    c[2].metric("Hit ratio", fmt(stats["hit_ratio"], "%", 0))
    c[3].metric("Sharpe (gross)", fmt(stats["sharpe"], "", 2))
    c[4].metric("Worst trade", fmt(stats["worst_trade_pts"], " pts"), stats["worst_trade_date"].strftime("%b %Y"), delta_color="off")
    c[5].metric("Max drawdown", fmt(stats["max_drawdown_pts"], " pts"))
    st.plotly_chart(charts.cumulative_pnl(trades), use_container_width=True)
    st.plotly_chart(charts.monthly_pnl_bars(trades), use_container_width=True)
    st.markdown(
        "**What this is.** Every 21 trading days, sell a one-month correlation swap at the COR1M strike and receive the realized "
        "correlation of the basket. P&L in correlation points per unit notional, gross of costs. A dispersion trade "
        "(short index variance, long single-name variances, vega-weighted) is this position plus a bet on single-name vol premia, "
        "so this isolates the correlation leg.\n\n"
        "**What it is not.** No bid-ask (several correlation points on a real swap), no margin, no vega-notional sizing by regime, "
        "and the basket membership is approximate (see the point-in-time note in the previous tab). The Sharpe is an upper bound, and the losses in "
        f"{stats['worst_trade_date'].strftime('%B %Y')} are what a real book would feel first."
    )
    st.markdown("#### Short correlation is a short put on the index")
    puts = cx.put_sale_pnl(px["SPY"], vix, trades.index)
    fit = cx.convexity_fit(trades, puts)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.convexity_scatter(fit), use_container_width=True)
    c2.plotly_chart(charts.corr_vs_put_paths(fit), use_container_width=True)
    down = fit["data"][fit["data"]["spx_ret_pct"] < -3]
    st.markdown(
        f"Fit of the monthly swap P&L on the S&P return: **{fit['slope_up']:+.2f} pts per 1%** of index move, plus **{fit['extra_slope_down']:+.2f} pts** "
        f"more per 1% on the way down (R² {fit['r2']:.2f}). Correlation with a short 15-delta put sold at the VIX: **{fit['corr_with_put']:+.2f}**. "
        f"In the {len(down)} months where the index fell more than 3%, the swap lost **{down['corr_pnl_pts'].mean():+.1f} pts** on average against "
        f"**{fit['avg_corr_pnl_all']:+.1f}** overall, and the put lost {down['put_pnl_pct'].mean():+.2f}% against {fit['avg_put_pnl_all']:+.2f}%. "
        "Selling correlation is selling crash insurance with a different label: price it, size it and hedge it like one."
    )
    with st.expander("Trade list"):
        st.dataframe((trades * 100).round(1), use_container_width=True)

    st.markdown("#### The full book: vega-weighted variance dispersion, rolled monthly")
    book, book_grid, book_surv = dispersion_book(offline)
    bk_net, _ = dbt.to_returns(book, "pnl_net")
    pn, pg = dbt.performance(bk_net), dbt.performance(dbt.to_returns(book, "pnl_gross")[0])
    ps = dbt.performance(dbt.to_returns(book_surv, "pnl_net")[0])
    c = st.columns(4)
    c[0].metric("Gross P&L / month", f"{book['pnl_gross'].mean():+.1f} vol pts", f"Sharpe {pg['sharpe']:.2f}", delta_color="off")
    c[1].metric("Net of costs", f"{book['pnl_net'].mean():+.1f} vol pts", f"Sharpe {pn['sharpe']:.2f}", delta_color="off")
    c[2].metric("Costs / month", f"{book['costs'].mean():.1f} vol pts", "0.5 index, 1.0 single names", delta_color="off")
    c[3].metric("Same book on today's top 50", f"Sharpe {ps['sharpe']:.2f}", "survivorship bias", delta_color="off")
    st.plotly_chart(charts.book_cumulative(book), use_container_width=True)
    st.dataframe(book_grid.style.format({"avg P&L (vol pts)": "{:+.2f}", "Sharpe": "{:.2f}", "hit %": "{:.0f}",
                                         "worst month at 10% vol, %": "{:.1f}", "max drawdown at 10% vol, %": "{:.1f}"}),
                 use_container_width=True, hide_index=True)
    st.markdown(
        "**What this is.** The book of the *Trade sheet* tab, rolled every 21 trading days since 2015: short a one-month index "
        "variance swap at the VIX, long one-month variance swaps on the point-in-time top 50 with vega notional w_i x V. Real "
        "variance-swap payoffs (convex, single names capped at 2.5x strike), half a bid-ask paid on every leg at every roll. "
        "Single-name strikes have no free history: they are rebuilt from VIX and COR1M (see *Where the premium comes from*), "
        "checked against the 27 Oct 2026 chains: correlation 0.89 across names, but the weighted level is 2.5 points too low "
        "(40.0 vs 42.5), which flatters the long legs. Treat the gross line as an upper bound.\n\n"
        "**Reading.** Gross, the book earns what the decomposition says it should: the correlation premium. Net, a vol point of "
        "bid-ask on 50 single-name swaps every month eats most or all of it, and the result swings with the single-name smile "
        "assumption (c_S), which no free data pins down. The same book on today's top 50 looks like a money machine: that is "
        "survivorship, not alpha. This matches the academic reading (Driessen, Maenhout and Vilkov, 2009): correlation risk is "
        "priced, and hard to harvest after costs. Returns at 10% vol use one constant sizing for the whole sample."
    )

# ---------- Where the premium comes from ----------
with tabs[5]:
    dec, rob = premium_decomposition(offline)
    pre = dcmp.summary(dec[dec.index < "2024-01-01"], "2015-01-01")
    post = dcmp.summary(dec, "2024-01-01")
    st.markdown(
        "**Question.** Sellers of S&P 500 variance (VIX² against what the index then realizes) earn a premium. "
        "Index variance is built from two inputs, single-name vols and their correlation, so the premium can be split "
        "exactly into what the market overcharged for each. Implied single-name variances have no free history: they are "
        "rebuilt from VIX and COR1M the way CBOE builds COR1M, then converted to variance-swap levels with the index and "
        "single-name smiles measured on the 50 live chains."
    )
    m = st.columns(4)
    m[0].metric("Premium 2015-2023", f"{pre['avg_total']:.0f} var pts", f"{pre['total_positive_%']:.0f}% of days positive", delta_color="off")
    m[1].metric("of which correlation", f"{pre['avg_corr']:+.0f}", f"{pre['corr_positive_%']:.0f}% of days positive", delta_color="off")
    m[2].metric("of which single-name vol", f"{pre['avg_single']:+.0f}", f"{pre['single_positive_%']:.0f}% of days positive", delta_color="off")
    m[3].metric("Since 2024: correlation / single", f"{post['avg_corr']:+.0f} / {post['avg_single']:+.0f}",
                f"premium {post['avg_total']:.0f}", delta_color="off")
    st.plotly_chart(charts.premium_split_by_year(dcmp.by_period(dec, start=str(pd.Timestamp(bt_start).year))), use_container_width=True)
    st.markdown(
        "**Reading.** The correlation part is positive in every year since 2007: the market charges for correlation, reliably. "
        "The single-name part swings around zero, positive in some years and negative in others: single-name variance is not reliably overpriced. "
        "This is the Driessen, Maenhout and Vilkov (2009) result, *the index variance premium is a correlation premium*, "
        "rebuilt here from free data. Since 2024 the correlation part has shrunk and the single-name part has grown: with COR1M "
        "at record lows, part of what index variance sellers earn now comes from single-name vol (AI megacaps, earnings)."
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.model_free_correlation(dec[dec.index >= pd.Timestamp(bt_start)]), use_container_width=True)
    c2.plotly_chart(charts.single_iv_vs_rv(dec[dec.index >= pd.Timestamp(bt_start)]), use_container_width=True)
    st.caption("Left: the model-free implied correlation sits above COR1M because index skew is steeper than single-name skew. "
               "Crash protection on the index is, for the most part, protection against correlation going to one.")
    legs = dcmp.monthly_legs(dec, start=str(pd.Timestamp(bt_start).date()))
    st.plotly_chart(charts.legs_cumulative(legs), use_container_width=True)
    ls = dcmp.leg_stats(legs)
    st.dataframe(ls.style.format({"avg": "{:.0f}", "vol": "{:.0f}", "sharpe": "{:.2f}", "hit_%": "{:.0f}", "worst": "{:.0f}",
                                  "share_of_tail_losses_%": "{:.0f}"}), use_container_width=True)
    st.markdown(
        "**Risk.** The two parts are the attribution of a short index variance position, not two separately tradable P&Ls "
        "(a real dispersion book also pays 50 bid-ask spreads). In the worst 5% of months for index variance sellers, most of the "
        "loss is attributed to single-name vols exploding, not to correlation: the correlation part is the steady earner."
    )
    st.subheader("Is it robust?")
    st.markdown(
        "Two choices drive the levels. **Calibration**: c_I is SPY at-the-money vol over SPY model-free vol (0.79 measured), "
        "c_S is single-name model-free over at-the-money vol (1.13 measured, 1.06 for the median name); both are treated as "
        "constants. **Basket**: the main case is the point-in-time top 50. The two fixed baskets are there to show what "
        "survivorship does: today's top 50 (high pre-megacap vol of NVDA, TSLA, AVGO... inflates the correlation part) and the "
        "31 names already in the top 50 in 2015 (no survivors added, but the names driving index variance after 2024 are missing)."
    )
    st.dataframe(rob.style.format({c: "{:.0f}" for c in rob.columns if c not in ("basket", "c_I", "c_S")} | {"c_I": "{:.2f}", "c_S": "{:.2f}"}),
                 use_container_width=True, hide_index=True)
    st.markdown(
        "What survives every row: the correlation part is positive before 2024, and after 2024 it falls while the single-name "
        "part rises. What does not: the size of that shift, and the sign of the single-name part. Report the direction, "
        "not a point estimate."
    )
    with st.expander("Method in four lines"):
        st.markdown(
            "1. $V(\\sigma,\\rho) = B + \\rho(A-B)$ with $A=(\\sum w_i\\sigma_i)^2$, $B=\\sum w_i^2\\sigma_i^2$.  \n"
            "2. Implied side at $t$: ATM single-name level solved from $c_I\\cdot$VIX and COR1M with the shape of 63-day realized vols, "
            "scaled by $c_S$ to variance swaps; model-free $\\rho_Q$ solved from VIX².  \n"
            "3. Realized side: the same quantities over $t+1 \\dots t+21$, index leg = SPY.  \n"
            "4. Shapley split: correlation part $= (\\rho_Q-\\rho_P)\\cdot\\tfrac12[(A_Q-B_Q)+(A_P-B_P)]$, single-name part = the rest. "
            "Validation on 27 Oct 2026: rebuilt average single-name variance-swap vol 40.7 against 41.6 replicated from the 49 chains; "
            "the naive version (COR1M plugged straight into VIX) gives 46.7."
        )

# ---------- Mechanics ----------
with tabs[6]:
    st.markdown(
        "The identity behind the trade, for weights $w_i$, vols $\\sigma_i$ and one average correlation $\\rho$:  \n"
        "$\\sigma_{index}^2 = \\rho\\,(\\sum_i w_i\\sigma_i)^2 + (1-\\rho)\\sum_i w_i^2\\sigma_i^2$  \n"
        "Index vol is capped by the weighted average of single vols ($\\rho = 1$). The gap is dispersion. "
        "Solve for $\\rho$ with implied vols and you have implied correlation; with realized vols, realized correlation."
    )
    ok = names.dropna(subset=["atm_iv"])
    st.plotly_chart(charts.index_vol_vs_rho(ok["atm_iv"].values, ok["weight"].values, summ["rho_implied"], summ["index_iv"]), use_container_width=True)
    st.caption("With today's single-name IVs, SPY would need a correlation of 100% to reach the weighted average single vol. "
               "The market sits near the bottom left: cheap index vol relative to its parts.")
    st.plotly_chart(charts.vol_decomposition(hist[hist.index >= pd.Timestamp(bt_start)], cor1m), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.corr_term_structure(cor1m[cor1m.index >= pd.Timestamp(bt_start)], cor3m[cor3m.index >= pd.Timestamp(bt_start)]), use_container_width=True)
    with c2:
        st.markdown("**Term structure.** COR3M above COR1M is the normal state: correlation, like vol, is expected to mean-revert up "
                    "from calm levels. An inverted correlation curve (1M above 3M) marks a shock in progress.")
        ratio = (cor1m / cor3m).dropna()
        st.metric("COR1M / COR3M today", fmt(ratio.iloc[-1], "", 2), "inverted" if ratio.iloc[-1] > 1 else "normal", delta_color="off")
    top20 = list(w["ticker"].head(20))
    st.plotly_chart(charts.heatmap(pairwise_correlation_matrix(px[top20], 63)), use_container_width=True)

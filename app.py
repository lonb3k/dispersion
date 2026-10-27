"""Equity dispersion: implied vs realized correlation on the S&P 500.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dispersion import backtest as bt, charts  # noqa: E402
from dispersion.correlation import pairwise_correlation_matrix, realized_correlation_history  # noqa: E402
from dispersion.data import load_cboe_index, load_prices  # noqa: E402
from dispersion.snapshot import implied_snapshot  # noqa: E402
from dispersion.universe import load_weights  # noqa: E402

st.set_page_config(page_title="Dispersion", page_icon="🔀", layout="wide")


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


def fmt(x, suffix="%", d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}{suffix}"


with st.sidebar:
    st.title("Dispersion")
    st.caption("Implied vs realized correlation on the S&P 500 top-50, the CBOE COR1M way.")
    top_n = st.slider("Basket size (largest by weight)", 20, 50, 50, step=5)
    offline = st.toggle("Offline (cached data only)", value=False)
    bt_start = st.date_input("Backtest from", pd.Timestamp("2015-01-01"))
    st.divider()
    st.caption("Prices: Yahoo Finance. Weights: SPY holdings (State Street). Implied correlation history: CBOE COR1M / COR3M. "
               "Single-name implied vols rebuilt from option mids with Black-76 on the put-call-parity forward.")

w = weights(top_n)
tickers = tuple(w["ticker"]) + ("SPY",)
px = prices(tickers, "2005-01-01", offline)
hist = realized_correlation_history(px[list(w["ticker"])], w.set_index("ticker")["weight"], index_prices=px["SPY"])
cor1m, cor3m, vix = cboe("COR1M", offline), cboe("COR3M", offline), cboe("VIX", offline)
crp_full = bt.correlation_risk_premium(cor1m, hist)

st.title("Equity dispersion: is correlation too expensive?")
st.caption(f"Basket: top {top_n} S&P 500 names covering {w['index_weight'].sum()*100:.0f}% of the index, re-weighted to 100%. "
           f"Last price {px.index[-1].date()}, last COR1M {cor1m.index[-1].date()}.")

tabs = st.tabs(["Today", "Implied vs realized", "Correlation swap backtest", "Mechanics"])

# ---------- Today ----------
with tabs[0]:
    target = st.select_slider("Option tenor (days)", [30, 60, 90], value=30)
    with st.spinner("Reading option chains name by name (about two minutes the first time)..."):
        names, summ = snapshot(top_n, target, offline)
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

# ---------- Implied vs realized ----------
with tabs[1]:
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
        st.markdown(
            "Implied correlation sits above realized almost all the time: index options are bid as portfolio insurance, "
            "single-name options are sold by overwriters, so the index-vs-singles spread prices a premium. "
            "The premium shrank sharply from 2024: the trade became crowded and implied correlation hit record lows."
        )
    st.warning(
        "Survivorship caveat: the realized basket is **today's** top names with fixed weights. Before 2015 several of them "
        "(NVDA, TSLA, AVGO, PLTR...) were small or not listed, so early realized correlation is understated and the premium "
        "overstated. The backtest start date in the sidebar defaults to 2015 for that reason. The CBOE rebalances its basket."
    )

# ---------- Backtest ----------
with tabs[2]:
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
        "and the basket is frozen (see the survivorship caveat). The Sharpe is an upper bound, and the losses in "
        f"{stats['worst_trade_date'].strftime('%B %Y')} are what a real book would feel first."
    )
    with st.expander("Trade list"):
        st.dataframe((trades * 100).round(1), use_container_width=True)

# ---------- Mechanics ----------
with tabs[3]:
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

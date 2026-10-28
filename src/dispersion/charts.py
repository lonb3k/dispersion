"""Plotly figures. One y-axis per chart, thin marks, categorical colors in fixed order."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Validated categorical palette (light surface), fixed slot order
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"], C["red"]]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, Helvetica, Arial, sans-serif", size=12, color=TEXT),
    margin=dict(l=48, r=24, t=84, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=MUTED)),
    xaxis=dict(showgrid=False, zeroline=False, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix="%"),
)


def _base(title: str, height: int = 380, **overrides) -> go.Figure:
    fig = go.Figure()
    layout = {**LAYOUT, **overrides}
    fig.update_layout(title=dict(text=title, x=0, y=0.98, yanchor="top", font=dict(size=14)), height=height, **layout)
    return fig


# ---------- time series ----------

def realized_vs_implied(rv: pd.DataFrame, implied: pd.Series | None, implied_name: str | None) -> go.Figure:
    fig = _base("Realized vol (annualized) vs 30d implied")
    for i, col in enumerate(rv.columns):
        fig.add_trace(go.Scatter(x=rv.index, y=rv[col], name=col, mode="lines",
                                 line=dict(width=1.5, color=SERIES[i]),
                                 hovertemplate="%{y:.1f}%"))
    if implied is not None and not implied.empty:
        fig.add_trace(go.Scatter(x=implied.index, y=implied, name=implied_name or "Implied", mode="lines",
                                 line=dict(width=2, color=C["red"]), hovertemplate="%{y:.1f}%"))
    return fig


# ---------- correlation history ----------

def implied_vs_realized(crp: pd.DataFrame) -> go.Figure:
    fig = _base("Implied correlation (CBOE COR1M) vs realized over the following month", height=400,
                yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix="%"))
    fig.add_trace(go.Scatter(x=crp.index, y=crp["implied"] * 100, name="Implied (COR1M)", mode="lines",
                             line=dict(width=1.8, color=C["red"]), hovertemplate="%{y:.1f}%"))
    fig.add_trace(go.Scatter(x=crp.index, y=crp["realized_fwd"] * 100, name="Realized next 21d (top-50 vs SPY)", mode="lines",
                             line=dict(width=1.3, color=C["blue"]), hovertemplate="%{y:.1f}%"))
    return fig


def premium_area(crp: pd.DataFrame) -> go.Figure:
    fig = _base("Correlation risk premium: implied minus realized, correlation points", height=300,
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix=""))
    p = crp["premium"] * 100
    fig.add_trace(go.Scatter(x=crp.index, y=p.clip(lower=0), name="Premium earned", mode="lines", fill="tozeroy",
                             line=dict(width=0.8, color=C["aqua"]), fillcolor="rgba(27,175,122,0.35)", hovertemplate="%{y:.1f} pts"))
    fig.add_trace(go.Scatter(x=crp.index, y=p.clip(upper=0), name="Premium lost", mode="lines", fill="tozeroy",
                             line=dict(width=0.8, color=C["red"]), fillcolor="rgba(227,73,72,0.35)", hovertemplate="%{y:.1f} pts"))
    return fig


def cumulative_pnl(trades: pd.DataFrame) -> go.Figure:
    fig = _base("Short correlation swap: cumulative P&L in correlation points (gross, 1 trade / month)", height=360,
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix=""))
    fig.add_trace(go.Scatter(x=trades.index, y=trades["cum_pnl"] * 100, name="Cumulative P&L", mode="lines",
                             line=dict(width=2, color=C["blue"]), hovertemplate="%{y:.0f} pts"))
    fig.add_trace(go.Scatter(x=trades.index, y=trades["drawdown"] * 100, name="Drawdown", mode="lines", fill="tozeroy",
                             line=dict(width=0.8, color=C["red"]), fillcolor="rgba(227,73,72,0.3)", hovertemplate="%{y:.0f} pts"))
    return fig


def monthly_pnl_bars(trades: pd.DataFrame) -> go.Figure:
    fig = _base("P&L per monthly trade, correlation points", height=300, hovermode="closest",
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix=""))
    p = trades["pnl"] * 100
    fig.add_trace(go.Bar(x=trades.index, y=p, name="Trade P&L",
                         marker=dict(color=[C["aqua"] if v >= 0 else C["red"] for v in p], line=dict(width=0)),
                         hovertemplate="%{x|%b %Y}: %{y:+.1f} pts"))
    fig.update_layout(showlegend=False, bargap=0.3)
    return fig


def vol_decomposition(hist: pd.DataFrame, cor: pd.Series) -> go.Figure:
    """Realized: index vol vs weighted average single vol. The gap is realized dispersion."""
    fig = _base("Realized: weighted average single-name vol vs index vol (21d)", height=360)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["avg_single_vol"], name="Avg single-name vol", mode="lines",
                             line=dict(width=1.5, color=C["orange"]), hovertemplate="%{y:.1f}%"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["index_vol"], name="Index vol (SPY)", mode="lines",
                             line=dict(width=1.5, color=C["blue"]), hovertemplate="%{y:.1f}%"))
    return fig


# ---------- today ----------

def names_iv_rv(names: pd.DataFrame) -> go.Figure:
    d = names.dropna(subset=["atm_iv"]).sort_values("iv_minus_rv", ascending=False)
    fig = _base("Per name: 30d implied minus 21d realized vol (positive = single-name vol rich)", height=420,
                hovermode="closest", xaxis=dict(showgrid=False, type="category"),
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix=" pts"))
    colors = [C["violet"] if pd.notna(e) else (C["aqua"] if v >= 0 else C["red"]) for v, e in zip(d["iv_minus_rv"], d["earnings"])]
    fig.add_trace(go.Bar(x=d["ticker"], y=d["iv_minus_rv"], marker=dict(color=colors, line=dict(width=0)), name="IV - RV",
                         customdata=d[["atm_iv", "rv21", "weight"]].values,
                         hovertemplate="%{x}: IV %{customdata[0]:.1f}%  RV %{customdata[1]:.1f}%  weight %{customdata[2]:.1%}"))
    fig.update_layout(showlegend=False, bargap=0.25)
    return fig


def index_vol_vs_rho(vols: np.ndarray, weights: np.ndarray, rho_now: float, index_iv: float) -> go.Figure:
    from .correlation import index_vol_from_correlation
    rhos = np.linspace(0, 1, 41)
    curve = [index_vol_from_correlation(r, vols, weights) for r in rhos]
    fig = _base("Index vol implied by the basket as a function of correlation (today's single-name IVs)", height=360,
                hovermode="closest", xaxis=dict(title="Average pairwise correlation", showgrid=False, tickformat=".0%"))
    fig.add_trace(go.Scatter(x=rhos, y=curve, mode="lines", name="sigma_index(rho)", line=dict(width=2, color=C["blue"]),
                             hovertemplate="rho %{x:.0%}: %{y:.1f}%"))
    fig.add_trace(go.Scatter(x=[rho_now], y=[index_iv], mode="markers", name="Market today",
                             marker=dict(size=12, color=C["red"], symbol="diamond"), hovertemplate="rho %{x:.1%}, SPY IV %{y:.1f}%"))
    return fig


def corr_term_structure(cor1m: pd.Series, cor3m: pd.Series) -> go.Figure:
    fig = _base("Implied correlation term structure: COR1M vs COR3M", height=320,
                yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix="%"))
    fig.add_trace(go.Scatter(x=cor1m.index, y=cor1m, name="COR1M", mode="lines", line=dict(width=1.5, color=C["blue"]), hovertemplate="%{y:.1f}%"))
    fig.add_trace(go.Scatter(x=cor3m.index, y=cor3m, name="COR3M", mode="lines", line=dict(width=1.5, color=C["orange"]), hovertemplate="%{y:.1f}%"))
    return fig


def heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = _base("Pairwise realized correlation, last 63 days, top 20 by weight", height=520, hovermode="closest",
                xaxis=dict(showgrid=False, tickangle=45), yaxis=dict(showgrid=False, autorange="reversed", ticksuffix=""))
    fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-0.2, zmax=1,
                             colorscale=[[0, "#fcfcfb"], [0.17, "#fcfcfb"], [1, SEQ[6]]],
                             colorbar=dict(title="rho", thickness=12), hovertemplate="%{y} / %{x}: %{z:.2f}"))
    fig.update_layout(margin=dict(l=60, r=24, t=60, b=80))
    return fig


# ---------- trade sheet ----------

def scenario_heatmap(grid: pd.DataFrame) -> go.Figure:
    fig = _base("Book P&L to expiry, vol points per unit index vega: realized correlation × single-name realization", height=420,
                hovermode="closest", xaxis=dict(title="k = single-name realized / implied", showgrid=False, type="category"),
                yaxis=dict(title="Realized correlation", showgrid=False, type="category", autorange="reversed", ticksuffix=""))
    z = grid.values.astype(float)
    lim = float(np.nanmax(np.abs(z)))
    fig.add_trace(go.Heatmap(z=z, x=list(grid.columns), y=list(grid.index), zmin=-lim, zmax=lim,
                             colorscale=[[0, C["red"]], [0.5, "#fcfcfb"], [1, C["aqua"]]],
                             text=np.round(z, 1), texttemplate="%{text:+.1f}", textfont=dict(size=11),
                             colorbar=dict(title="pts", thickness=12), hovertemplate="%{y}, %{x}: %{z:+.1f} pts"))
    return fig


def legs_bar(legs: pd.DataFrame) -> go.Figure:
    d = legs.sort_values("vega_weight", ascending=False).head(30)
    fig = _base("Long single-name variance legs: vega weight (bar) and IV − trailing RV (color)", height=380, hovermode="closest",
                xaxis=dict(showgrid=False, type="category"), yaxis=dict(gridcolor=GRID, zeroline=False, tickformat=".0%", ticksuffix=""))
    colors = [C["aqua"] if v >= 0 else C["red"] for v in d["iv_minus_rv"]]
    fig.add_trace(go.Bar(x=d["ticker"], y=d["vega_weight"], marker=dict(color=colors, line=dict(width=0)), name="vega weight",
                         customdata=d[["atm_iv", "rv21", "iv_minus_rv"]].values,
                         hovertemplate="%{x}: vega %{y:.1%}  IV %{customdata[0]:.1f}%  RV %{customdata[1]:.1f}%  (%{customdata[2]:+.1f})"))
    fig.update_layout(showlegend=False, bargap=0.25)
    return fig


# ---------- factors ----------

def group_corr_lines(gc: pd.DataFrame) -> go.Figure:
    fig = _base("Average pairwise realized correlation, 63d rolling", height=360,
                yaxis=dict(gridcolor=GRID, zeroline=False, tickformat=".0%", ticksuffix=""))
    for i, (col, name) in enumerate([("within_mag7", "Within Mag 7"), ("within_rest", "Within the other names"),
                                     ("cross", "Mag 7 vs the rest"), ("all", "All pairs")]):
        fig.add_trace(go.Scatter(x=gc.index, y=gc[col], name=name, mode="lines", line=dict(width=1.5, color=SERIES[i]),
                                 hovertemplate="%{y:.0%}"))
    return fig


def variance_share_lines(vs: pd.DataFrame) -> go.Figure:
    fig = _base("Mag 7: share of basket variance vs share of basket weight", height=320,
                yaxis=dict(gridcolor=GRID, zeroline=False, tickformat=".0%", ticksuffix=""))
    fig.add_trace(go.Scatter(x=vs.index, y=vs["mag7_variance_share"], name="Share of variance", mode="lines",
                             line=dict(width=2, color=C["orange"]), hovertemplate="%{y:.0%}"))
    fig.add_trace(go.Scatter(x=vs.index, y=vs["mag7_weight"], name="Share of weight (today's, fixed)", mode="lines",
                             line=dict(width=1.5, color=C["blue"], dash="dot"), hovertemplate="%{y:.0%}"))
    return fig


def factor_vs_implied(al: pd.DataFrame) -> go.Figure:
    fig = _base("First principal component share of variance (252d) vs implied correlation (COR1M)", height=340,
                yaxis=dict(gridcolor=GRID, zeroline=False, tickformat=".0%", ticksuffix=""))
    fig.add_trace(go.Scatter(x=al.index, y=al["factor"], name="PC1 share", mode="lines", line=dict(width=1.8, color=C["blue"]), hovertemplate="%{y:.0%}"))
    fig.add_trace(go.Scatter(x=al.index, y=al["implied"], name="COR1M", mode="lines", line=dict(width=1.3, color=C["red"]), hovertemplate="%{y:.0%}"))
    return fig


# ---------- convexity ----------

def convexity_scatter(fit: dict) -> go.Figure:
    d = fit["data"].sort_values("spx_ret_pct")
    fig = _base("Monthly short-correlation P&L vs S&P return over the same month", height=380, hovermode="closest",
                xaxis=dict(title="S&P 500 return over the trade (%)", showgrid=False, ticksuffix="%"),
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix=" pts", title="Correlation swap P&L"))
    fig.add_trace(go.Scatter(x=d["spx_ret_pct"], y=d["corr_pnl_pts"], mode="markers", name="trades",
                             marker=dict(size=6, color=C["blue"], opacity=0.5), customdata=d.index.strftime("%b %Y"),
                             hovertemplate="%{customdata}: %{y:+.1f} pts at %{x:+.1f}%"))
    r = d["spx_ret_pct"].values
    line = fit["intercept"] + fit["slope_up"] * r + fit["extra_slope_down"] * np.minimum(r, 0)
    fig.add_trace(go.Scatter(x=r, y=line, mode="lines", name="fit: a + b·r + c·min(r,0)", line=dict(width=2, color=C["red"]), hoverinfo="skip"))
    return fig


def corr_vs_put_paths(fit: dict) -> go.Figure:
    d = fit["data"]
    fig = _base("Cumulative P&L, both scaled to unit standard deviation: correlation swap vs short 15-delta put", height=340,
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix=""))
    a = (d["corr_pnl_pts"] / d["corr_pnl_pts"].std()).cumsum()
    b = (d["put_pnl_pct"] / d["put_pnl_pct"].std()).cumsum()
    fig.add_trace(go.Scatter(x=d.index, y=a, name="Short correlation", mode="lines", line=dict(width=1.8, color=C["blue"]), hovertemplate="%{y:.1f}"))
    fig.add_trace(go.Scatter(x=d.index, y=b, name="Short 15d put", mode="lines", line=dict(width=1.8, color=C["orange"]), hovertemplate="%{y:.1f}"))
    return fig


# ---------- variance premium decomposition ----------

def premium_split_by_year(by_year: pd.DataFrame) -> go.Figure:
    fig = _base("Index variance premium by year: what correlation and single-name vol contributed", height=420,
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix=""), barmode="relative", bargap=0.25)
    x = by_year.index.astype(str)
    fig.add_trace(go.Bar(x=x, y=by_year["corr_part"], name="Correlation part", marker=dict(color=C["blue"], line=dict(width=0)),
                         hovertemplate="%{y:.0f} var pts"))
    fig.add_trace(go.Bar(x=x, y=by_year["single_part"], name="Single-name vol part", marker=dict(color=C["orange"], line=dict(width=0)),
                         hovertemplate="%{y:.0f} var pts"))
    fig.add_trace(go.Scatter(x=x, y=by_year["total"], name="Total premium (VIX² − realized)", mode="markers",
                             marker=dict(symbol="diamond", size=9, color=TEXT, line=dict(width=2, color="white")),
                             hovertemplate="%{y:.0f} var pts"))
    fig.update_yaxes(title_text="variance points (vol² , annualized)")
    return fig


def legs_cumulative(legs: pd.DataFrame) -> go.Figure:
    fig = _base("Monthly short positions, cumulative P&L in variance points", height=400,
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix=""))
    for i, (c, name) in enumerate([("cum_total", "Short index variance"), ("cum_corr_part", "Short correlation part"),
                                   ("cum_single_part", "Short single-name vol part")]):
        fig.add_trace(go.Scatter(x=legs.index, y=legs[c], name=name, mode="lines",
                                 line=dict(width=2 if i == 0 else 1.6, color=[TEXT, C["blue"], C["orange"]][i]),
                                 hovertemplate="%{y:.0f}"))
    return fig


def model_free_correlation(dec: pd.DataFrame, smooth: int = 63) -> go.Figure:
    fig = _base(f"Correlation: model-free, COR1M, realized ({smooth}d avg)", height=400)
    d = dec.rolling(smooth).mean()
    for i, (c, name) in enumerate([("rho_q", "Model-free implied (variance swaps)"), ("rho_atm", "COR1M (at-the-money)"),
                                   ("rho_p", "Realized next 21d")]):
        fig.add_trace(go.Scatter(x=d.index, y=d[c] * 100, name=name, mode="lines",
                                 line=dict(width=1.8, color=[C["red"], C["violet"], C["blue"]][i]), hovertemplate="%{y:.1f}%"))
    return fig


def single_iv_vs_rv(dec: pd.DataFrame, smooth: int = 63) -> go.Figure:
    fig = _base(f"Single-name vol: implied vs realized ({smooth}d avg)", height=400)
    d = dec.rolling(smooth).mean()
    fig.add_trace(go.Scatter(x=d.index, y=d["avg_single_iv"], name="Implied, rebuilt", mode="lines",
                             line=dict(width=1.8, color=C["orange"]), hovertemplate="%{y:.1f}%"))
    fig.add_trace(go.Scatter(x=d.index, y=d["avg_single_rv_fwd"], name="Realized next 21d", mode="lines",
                             line=dict(width=1.6, color=C["blue"]), hovertemplate="%{y:.1f}%"))
    return fig

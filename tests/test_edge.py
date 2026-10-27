"""Trade sheet scenarios, factor decomposition and the convexity fit on synthetic inputs."""
import numpy as np
import pandas as pd

from dispersion import convexity as cx, factors as fa, tradesheet as ts


def _legs():
    return pd.DataFrame({"ticker": ["A", "B", "C", "NVDA"], "weight": [0.3, 0.3, 0.2, 0.2], "atm_iv": [30.0, 40.0, 25.0, 50.0],
                         "rv21": [28.0, 45.0, 20.0, 50.0], "earnings": [pd.NaT, pd.Timestamp("2030-01-01"), pd.NaT, pd.NaT]})


def test_book_is_flat_at_implied_correlation_and_k_one():
    legs, book = ts.build_book(_legs(), index_iv=20.0)
    assert abs(ts.scenario_pnl(legs, 20.0, book["rho_implied"], 1.0)) < 1e-9
    assert abs(ts.breakeven_correlation(legs, 20.0, 1.0) - book["rho_implied"]) < 1e-6


def test_higher_correlation_loses_and_more_single_realization_wins():
    legs, book = ts.build_book(_legs(), index_iv=20.0)
    r = book["rho_implied"]
    assert ts.scenario_pnl(legs, 20.0, r + 0.2, 1.0) < 0
    assert ts.scenario_pnl(legs, 20.0, r, 1.2) > 0
    grid = ts.scenario_grid(legs, 20.0)
    assert (grid.diff().dropna() <= 1e-9).all().all()  # monotone down the correlation axis


def test_exclude_earnings_drops_legs_and_renormalizes():
    legs, book = ts.build_book(_legs(), 20.0, exclude_earnings=True)
    assert book["legs"] == 3 and abs(legs["vega_weight"].sum() - 1) < 1e-12
    assert book["mag7_vega_share"] > 0.2


def _one_factor(n_days=600, n=12, rho=0.5, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n_days)
    e = rng.standard_normal((n_days, n))
    r = (np.sqrt(rho) * f[:, None] + np.sqrt(1 - rho) * e) * 0.01
    cols = ["NVDA", "AAPL", "MSFT"] + [f"S{i}" for i in range(n - 3)]
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=pd.bdate_range("2020-01-01", periods=n_days), columns=cols)


def test_first_factor_share_matches_one_factor_model():
    px = _one_factor()
    share = fa.first_factor_share(px, window=400, step=50)
    # in a one-factor model with n names the first eigenvalue of the correlation matrix is 1 + (n-1) rho
    expected = (1 + 11 * 0.5) / 12
    assert abs(share.mean() - expected) < 0.05


def test_group_correlations_are_all_similar_in_one_factor_model():
    gc = fa.group_correlations(_one_factor(), window=252).dropna()
    assert abs(gc["within_mag7"].mean() - gc["within_rest"].mean()) < 0.12
    assert gc["all"].between(0.3, 0.7).all()


def test_put_pnl_and_convexity_fit():
    idx = pd.bdate_range("2021-01-01", periods=400)
    spx = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 400))), index=idx)
    vix = pd.Series(18.0, index=idx)
    trade_dates = idx[::21][:-2]
    puts = cx.put_sale_pnl(spx, vix, trade_dates)
    assert (puts["put_premium_pct"] > 0).all() and (puts["put_pnl_pct"] <= puts["put_premium_pct"] + 1e-12).all()
    assert (puts["strike_pct"] < 100).all()
    trades = pd.DataFrame({"pnl": 0.05 + 0.01 * puts["spx_ret_pct"].values}, index=puts.index)
    fit = cx.convexity_fit(trades, puts)
    assert abs(fit["slope_up"] - 1.0) < 1e-6 and abs(fit["extra_slope_down"]) < 1e-6 and fit["r2"] > 0.999

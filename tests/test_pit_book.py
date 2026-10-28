"""Point-in-time membership and the variance dispersion book backtest."""
import numpy as np
import pandas as pd

from dispersion import dispersion_backtest as db
from dispersion.universe import point_in_time_weights
from test_decomposition import _market


def test_point_in_time_weights_pick_the_largest_caps_and_apply_next_day():
    idx = pd.bdate_range("2020-01-01", periods=60)
    close = pd.DataFrame({"A": np.linspace(10, 30, 60), "B": 20.0, "C": np.linspace(30, 10, 60)}, index=idx)
    shares = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
    w = point_in_time_weights(close, shares, top_n=2)
    jan_end = close.loc["2020-01"].index[-1]
    first_feb = close.loc["2020-02"].index[0]
    assert (w.loc[:jan_end].sum(axis=1) == 0).all()                  # nothing is known before the first month end
    assert np.allclose(w.loc[first_feb:].sum(axis=1), 1.0)
    caps_jan = close.loc[jan_end]
    assert set(w.loc[first_feb][w.loc[first_feb] > 0].index) == set(caps_jan.nlargest(2).index)  # decided on Jan 31, used from Feb 3
    last = w.iloc[-1]
    assert last["C"] == 0 and last["A"] > 0 and last["B"] > 0          # C shrank out of the top 2


def _inputs():
    px, w, spy = _market(n_days=900)
    ri = np.log(spy / spy.shift(1))
    vix = np.sqrt(ri.rolling(21).var() * 252 * 1e4).bfill() * 1.1
    cor = pd.Series(40.0, index=px.index)
    return px, w, spy, vix, cor


def test_costs_only_lower_the_book_and_legs_add_up():
    px, w, spy, vix, cor = _inputs()
    t0 = db.backtest(px, w, spy, vix, cor, start="2015-01-01", index_cost=0.0, single_cost=0.0)
    t1 = db.backtest(px, w, spy, vix, cor, start="2015-01-01", index_cost=0.5, single_cost=1.0)
    assert np.allclose(t0["pnl_gross"], t0["index_leg"] + t0["single_leg"])
    assert np.allclose(t0["pnl_net"], t0["pnl_gross"])
    assert (t1["pnl_net"] < t1["pnl_gross"]).all()


def test_variance_swap_payoff_is_zero_at_the_strike_and_convex():
    K = 20.0
    payoff = lambda rv: 1 / (2 * K) * (rv**2 - K**2)
    assert payoff(K) == 0
    assert payoff(K + 5) > -payoff(K - 5) > 0                         # convex: gains on the way up exceed losses on the way down
    assert abs(payoff(K + 0.01) - 0.01) < 1e-3                         # one vega point per vol point near the strike


def test_sizing_hits_the_target_vol():
    t = pd.DataFrame({"pnl_net": np.random.default_rng(1).normal(1.0, 4.0, 120)})
    r, size = db.to_returns(t, "pnl_net", target_vol=0.10)
    assert abs(r.std() * np.sqrt(12) - 0.10) < 1e-9 and size > 0

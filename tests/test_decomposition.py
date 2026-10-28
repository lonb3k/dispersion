"""Variance premium decomposition: identities and signs on a synthetic one-factor market."""
import numpy as np
import pandas as pd

from dispersion import decomposition as dc


def _market(n_days=900, n=10, rho=0.4, seed=1):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n_days)
    e = rng.standard_normal((n_days, n))
    vols = np.linspace(0.01, 0.025, n)
    r = (np.sqrt(rho) * f[:, None] + np.sqrt(1 - rho) * e) * vols
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    cols = [f"S{i}" for i in range(n)]
    px = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)
    w = pd.Series(np.linspace(2, 1, n), index=cols)
    w = w / w.sum()
    spy = pd.Series(100 * np.exp(np.cumsum((r * w.values).sum(axis=1))), index=idx)
    return px, w, spy


def test_moments_satisfy_the_basket_identity_and_recover_correlation():
    px, w, spy = _market()
    m = dc.basket_moments(px, w, spy, 252)
    assert np.allclose(m["idx_var"], m["B"] + m["rho"] * (m["A"] - m["B"]))
    assert abs(m["rho"].mean() - 0.4) < 0.08


def test_implied_level_reproduces_vix_exactly():
    px, w, spy = _market()
    shape = dc.basket_moments(px, w, spy, 63)
    vix = pd.Series(20.0, index=shape.index)
    cor = pd.Series(30.0, index=shape.index)
    q = dc.implied_moments(vix, cor, shape, 1.0, 1.0)
    assert np.allclose(q["B_q"] + q["rho_q"] * (q["A_q"] - q["B_q"]), 400.0)
    assert np.allclose(q["A_q"] / q["B_q"], (shape["A"] / shape["B"]).reindex(q.index))


def _fair(px, w, spy, horizon=21):
    """VIX and COR equal to what the market then realizes: no premium at all."""
    fwd = dc.basket_moments(px, w, spy, horizon).shift(-horizon).dropna()
    return np.sqrt(fwd["idx_var"]), fwd["rho"] * 100


def test_no_premium_when_implied_equals_realized():
    px, w, spy = _market()
    vix, cor = _fair(px, w, spy)
    dec = dc.decompose(px, w, spy, vix, cor, c_index_atm=1.0, c_single_vs=1.0)
    assert np.allclose(dec["total"], 0, atol=1e-8)
    assert np.allclose(dec["corr_part"], 0, atol=1e-8)
    assert np.allclose(dec["single_part"], 0, atol=1e-8)


def test_overpriced_correlation_shows_up_in_the_correlation_part():
    px, w, spy = _market()
    vix, cor = _fair(px, w, spy)
    dec = dc.decompose(px, w, spy, vix, cor + 15, c_index_atm=1.0, c_single_vs=1.0)  # same index variance, correlation priced 15 pts too high
    assert np.allclose(dec["total"], 0, atol=1e-8)
    assert (dec["corr_part"] > 0).all() and (dec["single_part"] < 0).all()


def test_shapley_split_matches_both_orderings():
    px, w, spy = _market()
    vix, cor = _fair(px, w, spy)
    dec = dc.decompose(px, w, spy, vix * 1.2, cor + 10)

    def V(A, B, r):
        return B + r * (A - B)

    a = V(dec["A_q"], dec["B_q"], dec["rho_q"]) - V(dec["A_q"], dec["B_q"], dec["rho_p"])
    b = V(dec["A_p"], dec["B_p"], dec["rho_q"]) - V(dec["A_p"], dec["B_p"], dec["rho_p"])
    assert np.allclose(dec["corr_part"], 0.5 * (a + b))
    assert np.allclose(dec["corr_part"] + dec["single_part"], dec["var_q"] - dec["var_p"])
    legs = dc.monthly_legs(dec, start=None)
    st = dc.leg_stats(legs)
    assert abs(st.loc["Short correlation", "share_of_tail_losses_%"] + st.loc["Short single-name variance", "share_of_tail_losses_%"] - 100) < 1e-6


def test_steeper_index_skew_lifts_model_free_correlation_above_the_atm_index():
    px, w, spy = _market()
    shape = dc.basket_moments(px, w, spy, 63)
    vix = pd.Series(20.0, index=shape.index)
    cor = pd.Series(30.0, index=shape.index)
    naive = dc.implied_moments(vix, cor, shape, 1.0, 1.0)
    q = dc.implied_moments(vix, cor, shape, 0.80, 1.10)
    assert np.allclose(naive["rho_q"], 0.30)
    assert (q["rho_q"] > q["rho_atm"]).all()
    assert np.allclose(q["B_q"] + q["rho_q"] * (q["A_q"] - q["B_q"]), 400.0)
    assert (q["A_q"] < naive["A_q"]).all()  # single-name variances no longer absorb the index skew

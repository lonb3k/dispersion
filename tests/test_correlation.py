"""The basket identity and the realized correlation estimator on simulated data."""
import numpy as np
import pandas as pd

from dispersion import backtest as bt
from dispersion.correlation import implied_correlation, index_vol_from_correlation, realized_correlation_history


def test_identity_round_trip():
    rng = np.random.default_rng(0)
    vols = rng.uniform(0.15, 0.6, 30)
    w = rng.uniform(0.5, 2, 30)
    for rho in (0.05, 0.3, 0.8):
        assert abs(implied_correlation(index_vol_from_correlation(rho, vols, w), vols, w) - rho) < 1e-10


def test_index_vol_bounded_by_average_single_vol():
    vols, w = np.array([0.2, 0.4, 0.3]), np.array([0.5, 0.3, 0.2])
    assert index_vol_from_correlation(1.0, vols, w) == (w * vols).sum()
    assert index_vol_from_correlation(0.0, vols, w) == np.sqrt((w**2 * vols**2).sum())


def _simulate(n_days: int, n_names: int, rho: float, seed: int = 1):
    """One-factor model: r_i = sqrt(rho) f + sqrt(1-rho) e_i, equal vols -> pairwise correlation rho."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n_days)
    e = rng.standard_normal((n_days, n_names))
    r = (np.sqrt(rho) * f[:, None] + np.sqrt(1 - rho) * e) * 0.01
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"S{i}" for i in range(n_names)])
    return prices


def test_realized_correlation_recovers_factor_correlation():
    rho = 0.4
    prices = _simulate(3000, 40, rho)
    w = pd.Series(1.0, index=prices.columns)
    hist = realized_correlation_history(prices, w, window=63)
    assert abs(hist["rho_basket"].mean() - rho) < 0.03
    # with the basket itself as index both conventions coincide
    assert np.allclose(hist["rho"], hist["rho_basket"])


def test_pre_ipo_names_are_renormalized():
    prices = _simulate(600, 10, 0.3)
    prices.iloc[:300, 0] = np.nan  # name 0 lists halfway
    w = pd.Series(1.0, index=prices.columns)
    hist = realized_correlation_history(prices, w, window=21)
    assert hist["n_names"].iloc[0] == 9 and hist["n_names"].iloc[-1] == 10
    # 21-day windows on 9-10 names are noisy day to day, but the average must sit near the factor correlation
    assert abs(hist["rho_basket"].mean() - 0.3) < 0.05
    assert abs(hist["rho_basket"].iloc[:280].mean() - hist["rho_basket"].iloc[300:].mean()) < 0.08


def test_backtest_pnl_is_strike_minus_realized():
    idx = pd.bdate_range("2020-01-01", periods=200)
    cor = pd.Series(40.0, index=idx)
    realized = pd.DataFrame({"rho": 0.25}, index=idx)
    crp = bt.correlation_risk_premium(cor, realized, horizon=21)
    trades = bt.correlation_swap_backtest(crp, horizon=21)
    assert np.allclose(trades["pnl"], 0.15)
    stats = bt.backtest_stats(trades)
    assert stats["hit_ratio"] == 100 and abs(stats["avg_pnl_pts"] - 15) < 1e-9

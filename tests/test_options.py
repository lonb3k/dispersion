"""The vendored Black-76 code: round trip and forward recovery."""
import numpy as np
import pandas as pd

from dispersion import options as o


def test_implied_vol_round_trip():
    F, t, disc = 100.0, 0.25, 0.99
    for K, sigma, is_call in [(90, 0.2, False), (100, 0.35, True), (115, 0.5, True)]:
        price = o.black76_price(F, K, t, sigma, is_call, disc)
        assert abs(o.implied_vol(price, F, K, t, is_call, disc) - sigma) < 1e-5


def test_atm_from_flat_chain():
    spot, t = 100.0, 30 / 365
    disc = np.exp(-o.RISK_FREE * t)
    F = 101.0
    rows = []
    for K in np.arange(75, 125, 1.0):
        for typ in ("call", "put"):
            p = o.black76_price(F, K, t, 0.25, typ == "call", disc)
            rows.append({"strike": K, "type": typ, "bid": p * 0.99, "ask": p * 1.01, "last": p, "iv_yahoo": np.nan, "volume": 1, "openInterest": 1})
    chain = pd.DataFrame(rows)
    cleaned, info = o.clean_chain(chain, spot, pd.Timestamp.today().normalize() + pd.Timedelta(days=30))
    assert abs(info["forward"] - F) < 0.05
    assert abs(o.skew_metrics(cleaned)["atm_iv"] - 25) < 0.05

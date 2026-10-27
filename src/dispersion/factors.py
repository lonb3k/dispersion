"""Why is implied correlation at a record low? Because the index became a few uncorrelated giants.

Two decompositions of the basket:
- Mag 7 vs the rest: realized correlation inside each group and across, and the share of index
  variance that comes from the mega-caps.
- Principal components: the share of return variance explained by the first factor, rolling.
  When the first factor explains 40% of variance the market trades as one; at 20% it trades as
  fifty stories. Implied correlation follows that share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAG7 = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA"]


def _returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def group_correlations(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Rolling average pairwise correlation within Mag 7, within the rest, and between the groups."""
    r = _returns(prices)
    m7 = [c for c in r.columns if c in MAG7]
    rest = [c for c in r.columns if c not in MAG7]
    out = {"within_mag7": [], "within_rest": [], "cross": [], "all": []}
    idx = []
    for end in range(window, len(r) + 1, 5):
        c = r.iloc[end - window:end].corr()
        a = c.values
        n = len(a)
        off = a[~np.eye(n, dtype=bool)]
        idx.append(r.index[end - 1])
        out["all"].append(np.nanmean(off))
        cm = c.loc[m7, m7].values
        out["within_mag7"].append(np.nanmean(cm[~np.eye(len(cm), dtype=bool)]) if len(cm) > 1 else np.nan)
        cr = c.loc[rest, rest].values
        out["within_rest"].append(np.nanmean(cr[~np.eye(len(cr), dtype=bool)]) if len(cr) > 1 else np.nan)
        out["cross"].append(np.nanmean(c.loc[m7, rest].values) if m7 and rest else np.nan)
    return pd.DataFrame(out, index=pd.DatetimeIndex(idx))


def variance_shares(prices: pd.DataFrame, weights: pd.Series, window: int = 63) -> pd.DataFrame:
    """Share of basket variance coming from Mag 7 (their own variances and their covariances with
    everything), rolling, with the basket vol."""
    r = _returns(prices)
    w = weights.reindex(r.columns).fillna(0)
    w = w / w.sum()
    m7 = [c for c in r.columns if c in MAG7]
    rows, idx = [], []
    for end in range(window, len(r) + 1, 5):
        cov = r.iloc[end - window:end].cov()
        total = float(w.values @ cov.values @ w.values)
        contrib = (cov @ w) * w  # each name's share of portfolio variance
        rows.append({"mag7_variance_share": float(contrib[m7].sum() / total), "mag7_weight": float(w[m7].sum()),
                     "basket_vol": float(np.sqrt(total * 252) * 100)})
        idx.append(r.index[end - 1])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def first_factor_share(prices: pd.DataFrame, window: int = 252, step: int = 5) -> pd.Series:
    """Rolling share of total return variance explained by the first principal component."""
    r = _returns(prices).dropna(how="all")
    vals, idx = [], []
    for end in range(window, len(r) + 1, step):
        x = r.iloc[end - window:end].dropna(axis=1)
        x = x - x.mean()
        c = np.corrcoef(x.values, rowvar=False)
        eig = np.linalg.eigvalsh(c)
        vals.append(float(eig[-1] / eig.sum()))
        idx.append(r.index[end - 1])
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name="pc1_share")


def align_with_implied(series: pd.Series, cor: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"factor": series}).join(cor.rename("implied") / 100, how="inner").dropna()
    return df

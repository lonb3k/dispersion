# Equity Dispersion: implied vs realized correlation on the S&P 500

**Live morning sheet: [https://morning-sheet-olive.vercel.app/dispersion](https://morning-sheet-olive.vercel.app/dispersion)**, regenerated every weekday after the US close.

![morning sheet](docs/morning_sheet.gif)

Streamlit app that rebuilds the CBOE implied correlation index from live option chains,
compares it with the realized correlation of the basket since 2006, and backtests the
correlation leg of a dispersion trade. Free data only, so anyone can rerun it.

Companion to the [volatility dashboard](../vol_dashboard): same option-pricing code
(Black-76 on the put-call-parity forward), applied to 50 single names at once.

## Results worth defending

**1. Implied correlation rebuilt from 50 option chains lands within 0.6 point of the CBOE print.**
On 2026-10-27, solving the basket identity with the 30-day ATM implied vols of the 50 largest
S&P 500 names and of SPY gives 10.0% implied correlation; the CBOE COR1M closed at 10.6%.
Same method as the CBOE (top 50, re-weighted), same answer.

**2. The famous correlation premium is mostly survivorship bias.**
Measure realized correlation on today's top 50 and implied correlation (COR1M) looks about 13
points too expensive since 2015: a short one-month correlation swap at the COR1M strike earns
+13.0 points a month, 87% of months, Sharpe 3.3. That basket is wrong in the past: today's winners
(NVDA, TSLA, AVGO, PLTR...) were smaller, more volatile and less correlated names, and the index's
actual top 50 held XOM, PG, VZ or Citi instead. Rebuilt point in time (each month, the 50 largest
names by today's float shares times that month's price), the two baskets share 47 names in 2026 and
realize the same correlation (9.0% vs 9.3%), but only 29 in 2015, and the fixed one shows 0.2%
realized correlation in 2006 against 26.7%. On the point-in-time basket:

| Since 2015 | Today's top 50 | Point-in-time top 50 |
|---|---|---|
| Average COR1M / realized next month | 28.9% / 15.9% | 28.6% / 28.4% |
| Short correlation swap, average P&L | +13.0 pts | +0.4 pts |
| Hit ratio / Sharpe (gross) | 87% / 3.3 | 57% / 0.1 |

At the money, implied correlation has been an almost unbiased forecast of next month's realized
correlation. It carries a premium only in stress (+7.9 points when VIX is above 30) and runs
below realized in calm markets (−2.8 points under VIX 15).

**3. The index variance premium is a correlation premium, and it sits in the skew.**
`decomposition.py` splits what sellers of index variance earn (VIX² against the variance the index
realizes over the next 21 days) into the part due to correlation and the part due to single-name vol,
with an exact Shapley split of the basket identity. Implied single-name variances have no free history,
so they are rebuilt from VIX and COR1M the way CBOE builds COR1M (at the money), then converted to
variance-swap levels with the index and single-name smiles measured on the 50 live chains
(SPY at-the-money / model-free vol 0.79, single names model-free / at-the-money 1.13). Plugging COR1M
straight into VIX, the naive route, overstates single-name implied vol by 12% against the chains.

| Point-in-time top 50, variance points | 2015-2023 | 2024-2026 |
|---|---|---|
| Index premium, VIX² − realized | 70 | 84 |
| of which correlation | +80 | +56 |
| of which single-name vol | −10 | +28 |

The correlation part is positive in every year since 2007; the single-name part swings around zero,
which is the Driessen, Maenhout and Vilkov (2009) result rebuilt with free data. Result 2 and this one
fit together: model-free implied correlation averages 38% since 2015, against 29% for COR1M and 28%
realized. The premium is not in at-the-money correlation, it is in index skew. Index puts are expensive
because they price correlation going to one in a crash.

Since 2024 the correlation part has fallen and the single-name part has risen in all nine
specifications tested (three calibrations of the smiles, three baskets); the size of the shift is not
robust, the direction is. In the worst 5% of months for index variance sellers, 70% to 100% of the
loss is attributed to single-name vols exploding.

**4. The full dispersion book, backtested: priced, but not harvestable after costs.**
`dispersion_backtest.py` rolls the vega-weighted variance dispersion book every month since 2015:
short a one-month index variance swap at the VIX, long one-month variance swaps on the point-in-time
top 50 with vega w_i x V, real variance-swap payoffs (single names capped at 2.5x strike), half a
bid-ask paid on every leg at every roll.

| Since 2015, per unit of index vega | Gross | Net (0.5 / 1 vol pt) |
|---|---|---|
| Average P&L per month | +1.0 vol pt | −0.4 vol pt |
| Sharpe | 1.18 | −0.47 |

Across single-name smile assumptions and cost levels, the net Sharpe runs from −2.8 to +0.9. The
same book on today's top 50 shows a Sharpe of 3.6: survivorship again. On the one day the rebuilt
single-name strikes can be checked against real chains, they are 2.5 points too low (40.0 vs 42.5),
which flatters the long legs, so the gross line is an upper bound. Correlation risk is priced and
hard to harvest after costs, which is the limits-to-arbitrage reading of Driessen et al.

**5. The book, priced today: flat at the implied correlation, dead in the crash corner.**
`tradesheet.py` builds the long-singles / short-index variance book from today's 50 chains and
sweeps realized correlation against how much single names realize relative to their implied.
At today's trailing vols the book would lose about 4 points: single-name vol is priced for
earnings that have not printed yet.

**6. Implied correlation is at record lows because the index became a few uncorrelated giants.**
`factors.py`: the Mag 7 are 54% of the basket's weight and 66% of its variance, their pairwise
correlation (20%) is three times that of the other names (7%), and the share of variance
explained by the first principal component fell from 40-45% in 2020-2022 to 20% today.
That share and COR1M move together (correlation +0.60 since 2015).

**7. Selling correlation is selling an index put.**
`convexity.py` regresses the monthly correlation-swap P&L on the S&P return of the same month:
+2.1 points per 1% of index move, 0.5 more on the way down (R² 0.46), correlation +0.43 with a short
15-delta put sold at the VIX. In the 21 months since 2015 where the index fell more than 3%, the swap
lost 17 points on average against +0.4 overall.

## What the app shows

| Tab | Content |
|---|---|
| **Trade sheet** | The vega-weighted book from today's chains, breakeven correlation, scenario grid (realized correlation × single-name realization), P&L at trailing vols, dollar view |
| **Why so low** | Mag 7 vs the rest: pairwise correlations, share of variance vs share of weight, first principal component share against COR1M |
| **Today** | Per-name 30/60/90d ATM implied vol vs 21d realized, earnings flags, our implied correlation next to the CBOE print, implied dispersion |
| **Implied vs realized** | COR1M vs realized correlation over the next 21 days since 2006, premium by year and by VIX regime |
| **Correlation swap backtest** | Monthly short correlation at the COR1M strike: cumulative P&L, drawdown, hit ratio, worst trades, the short-put comparison, and the full vega-weighted variance dispersion book rolled monthly, gross and net, with its sensitivity grid |
| **Where the premium comes from** | The index variance premium split into correlation and single-name vol by year, model-free vs ATM implied correlation, rebuilt single-name implied vol vs realized, cumulative P&L of the three short positions, robustness across calibrations and baskets |
| **Mechanics** | Index vol as a function of correlation with today's single-name IVs, realized index vol vs average single vol, COR1M vs COR3M term structure, pairwise correlation heatmap |

## The identity

For weights w_i, vols s_i and one average pairwise correlation rho:

    sigma_index^2 = rho * (sum_i w_i s_i)^2 + (1 - rho) * sum_i w_i^2 s_i^2

Index vol is bounded by the weighted average of single vols (rho = 1). Solve for rho with
implied vols and you have implied correlation; with realized vols, realized correlation.
A dispersion trade (short index variance, long single-name variances, vega-weighted) is a
short position in rho plus a bet on single-name variance premia. The backtest isolates the
rho leg with a correlation swap.

## Decisions, and why

- **Top 50, re-weighted, SPY as the index leg.** That is the CBOE COR methodology, so our number
  is comparable to COR1M. Using the top-50 basket itself as the index gives a different, higher
  correlation (pure pairwise inside the basket); both are computed, `rho` and `rho_basket`.
- **Implied vols rebuilt from mids** with Black-76 on the implied forward, not Yahoo's IV field,
  which ignores the forward and breaks put-call parity.
- **ATM vol by interpolation at the forward** across the put/call boundary, so the number
  does not depend on which side quotes.
- **Earnings flags.** Single-name vol is bid into results; 40 of 50 names reporting within the
  tenor is why implied correlation is seasonally low in late October and late January.
- **Names without history drop out and weights renormalize** on each date, so the basket
  does not shrink artificially before an IPO.
- **Point-in-time basket for everything historical.** Each month, the 50 largest names among today's
  250 largest by today's float-adjusted share count (from the SPY holdings file) times that month's
  split-adjusted, not dividend-adjusted, price. Known gaps: share counts move (Citi's 2008-2009 dilution
  makes it look too big before, Apple's buybacks too small), and names delisted since (Celgene, EMC...)
  cannot enter. The sidebar switches back to the fixed basket of today's names to show the bias.
- **Non-overlapping monthly trades** in the backtest so the hit ratio and Sharpe are not
  inflated by daily overlapping observations.

- **Scenario grid in two dimensions**, not a single "what if correlation goes to X": a crash raises
  single-name vol and correlation together, and the book's loss depends on both.
- **Put sold at the VIX without skew.** The comparison is about the shape of the payoff, not the
  level of the premium; adding skew would only make the put richer.

## Tests

```bash
.venv/bin/python -m pytest -q tests
```

Twenty-five tests. Among them: point-in-time weights pick the largest caps and apply from the next trading day; the book's legs add up and costs only lower it; variance-swap payoffs are zero at the strike and convex; the variance premium decomposition is exactly zero when implied equals realized, puts an overpriced correlation in the correlation part, and matches both orderings of the Shapley split; the identity round-trips for any rho; index vol equals the weighted average single
vol at rho = 1; the realized estimator recovers the factor correlation of a simulated one-factor
market; names listing mid-sample are handled; the backtest P&L equals strike minus realized;
Black-76 round-trips price to IV and recovers the forward and ATM vol from a synthetic chain.

## Daily snapshot

`scripts/snapshot_implied_corr.py` stores our implied correlation at 30 and 90 days with the
CBOE print next to it, in `data/implied_corr_history.csv`. `.github/workflows/snapshot.yml`
runs it on weekdays after the US close. Over time this gives a tracking record of the
replication and a 90-day implied correlation that the CBOE does not publish.

## Data

- Prices: Yahoo Finance via `yfinance`, mirrored to `data/prices.csv`.
- Weights: SPY holdings file from State Street, `data/spy_holdings.xlsx`, top 50 to `data/weights_top50.csv`.
- Point-in-time universe: closes of today's 250 largest names since 2005, split-adjusted (`data/pit_close.csv.gz`)
  and total-return (`data/pit_adj.csv.gz`), with float-adjusted shares in `data/pit_shares.csv`.
- Implied correlation history: CBOE public CSV endpoint for COR1M, COR3M and VIX, since 2006.
- Option chains: Yahoo, delayed. Outside US hours most quotes fall back to last trades and the app says so.


## Morning sheet

`src/dispersion/sheet.py` reads the dashboard for you: it runs the functions above on the
latest close and returns a desk note as a plain dict (tiles with sparklines, sentences with the
numbers filled in, trade ideas ranked by historical conditional expectation divided by the
historical worst case in the same regime, and the Plotly figures as JSON). It covers COR1M against our 50-chain implied correlation and realized correlation, the vega-weighted book at trailing vols, and a ranked correlation-swap / dispersion-book / crash-hedge call with conditional expectation and worst case.

- `python scripts/build_sheet.py [--offline]` writes `sheet/latest.json`.
- The daily GitHub Actions workflow rebuilds it after the close and commits it.
- The front end ([lonb3k/morning-sheet](https://github.com/lonb3k/morning-sheet), Next.js on Vercel)
  reads that JSON and renders it at [https://morning-sheet-olive.vercel.app/dispersion](https://morning-sheet-olive.vercel.app/dispersion), next to the sheets of the
  two sister projects. The Streamlit app stays the full interactive version.

No financial logic lives in `sheet.py`: every number comes from a function that already exists in `src/`.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Layout

```
app.py                         Streamlit UI
src/dispersion/
  universe.py                  SPY holdings, top-N weights
  data.py                      prices, CBOE indices, option chains, earnings dates
  options.py                   Black-76, implied forward, IV inversion, ATM / skew quotes
  correlation.py               the identity, realized correlation history, pairwise matrix
  snapshot.py                  today's implied correlation from 50 chains
  backtest.py                  correlation risk premium, correlation swap backtest, regime tables
  tradesheet.py                vega-weighted book, scenario grid, breakeven correlation
  factors.py                   Mag 7 vs rest correlations, variance shares, first principal component
  convexity.py                 short put simulation, convexity fit of the swap P&L
  decomposition.py             index variance premium = correlation part + single-name part, robustness
  dispersion_backtest.py       vega-weighted variance dispersion book rolled monthly, strike check, sizing
  charts.py                    Plotly figures
scripts/snapshot_implied_corr.py
tests/
data/
```

## Next

- Variance dispersion P&L with actual single-name variance strikes, once the daily snapshot has
  built a history of single-name implied vols.
- Sector dispersion (tech vs the rest) and the contribution of each name to the index-vs-singles gap.
- Vol-targeted basket optimizer on the same universe.

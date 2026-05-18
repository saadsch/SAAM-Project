# Part 3 - Interpretation notes

Portfolio summary (annualised, RF = 1.75% from `data/processed/rf_rate.csv`,
implementation 2014-01 - 2025-12; full table in
`outputs/tables/part3_summary_*_carbon50.csv`).

| Portfolio | Ann. return | Ann. vol | Sharpe (excess) | TE vs VW | Cumulative |
|---|---:|---:|---:|---:|---:|
| VW (P_vw)              | 6.72% | 12.88% | 0.386 | -    | +18.3% |
| MV (P_mv_oos)          | 8.30% | 11.58% | 0.565 | 9.07% | +60.3% |
| MV(-50%) (P_mv_oos(0.5))| 9.06% | 11.70% | 0.624 | 9.01% | +83.1% |
| VW(-50%) (P_vw_oos(0.5))| 7.79% | 13.59% | 0.444 | 3.24% | +46.0% |

## P_mv_oos vs P_mv_oos(0.5)

- Imposing a 50% cut on `CF(P_mv_oos)` does NOT damage performance: annualised
  return rises from 8.30% to 9.06% and Sharpe from 0.565 to 0.624.
- Volatility is virtually unchanged (11.58% -> 11.70%); the green MV portfolio
  stays inside the same low-vol corner of the universe.
- Tracking error vs VW is essentially identical (9.07% -> 9.01%): both MV
  variants are far from VW by construction (small concentrated long-only).
- Per-year CF check (`part3_constraint_slack_mv_carbon50.csv`): the 50% cap is
  binding (`slack ~ 0`) every single year. The optimiser is reallocating
  weight from carbon-intensive low-vol names towards the next cheapest set.

## P_vw vs P_vw_oos(0.5)

- The 50%-cap TE portfolio strictly dominates VW in return (+1.07 pp) and
  Sharpe (+0.058), and it lifts cumulative wealth from +18% to +46% over 12
  years.
- Volatility rises only modestly (12.88% -> 13.59%); the realised tracking
  error annualised is 3.24%, which is a small active risk budget.
- `part3_constraint_slack_vw_carbon50.csv` shows the cap binding every year:
  the green VW(-50%) portfolio sits exactly on the 50%-of-CF(P_vw) boundary.

## Financial trade-off

- For both MV and VW reference points, halving the carbon footprint did
  not come at a financial cost over the 2014-2025 sample - on this Pacific
  universe, the high-CF firms underperformed and removing them helped both
  return and risk-adjusted return.

## Carbon footprint reduction achieved

- MV(-50%) carbon footprint averages roughly 50% of MV's by construction,
  binding every year (see `part3_cf_mv_vs_mv_carbon50.png`).
- VW(-50%) likewise sits at exactly 50% of CF(P_vw) each year
  (`part3_cf_vw_vs_vw_carbon50.png`).
- WACI follows the same pattern but is not directly constrained
  (`part3_waci_*.png`); it tends to drop because high-CF firms are also
  high-CI in this sample.

## Tracking-error effect

- VW(-50%): annualised TE ~3.24% (ex-post) and 3.0-3.7% (ex-ante per year,
  `part3_tracking_error_vw_carbon50.png`). Reasonable for a passive investor
  with a soft sustainability mandate.
- MV(-50%): ~9.0% TE vs VW. This is *not* the optimisation objective; MV is
  a pure low-vol target so distance from VW is structural.

## Concentration / tilt

- Top-10 contributors (`part3_top10_cf_contributors.csv`,
  `part3_top10_waci_contributors.csv`) are dominated each year by AU and JP
  utility / materials names (e.g. Rio Tinto, BHP, JFE, Nippon Steel, Tepco
  in early years, gradually replaced by mining majors).
- `outputs/tables/part3_weights_*.csv` show MV-style strategies concentrating
  ~5-15 names; VW(-50%) keeps a broad universe (200+ names) but reweights
  away from the worst CF tail.

## Feasibility

- All 12 allocation years (2013-2024) optimised successfully for every
  Part 3 strategy. Min carbon-constraint slack ~ -1e-13 (numerical noise).
  No infeasibility encountered in the eligible universe.

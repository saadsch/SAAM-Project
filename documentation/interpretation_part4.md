# Part 4 - Interpretation notes (Net-Zero trajectory)

Portfolio summary (annualised, RF = 1.75%, implementation 2014-01 - 2025-12;
full table in `outputs/tables/part4_summary_vw_vs_carbon50_vs_netzero.csv`).

| Portfolio | Ann. return | Ann. vol | Sharpe (excess) | TE vs VW | Cumulative |
|---|---:|---:|---:|---:|---:|
| VW (P_vw)                | 6.72% | 12.88% | 0.386 | -    | +18.3% |
| VW(-50%) (P_vw_oos(0.5)) | 7.79% | 13.59% | 0.444 | 3.24% | +46.0% |
| VW(NZ) (P_vw_oos(NZ))    | 7.70% | 13.63% | 0.437 | 3.16% | +43.6% |

## P_vw vs P_vw_oos(0.5) vs P_vw_oos(NZ)

- Both decarbonised TE portfolios outperform VW on raw and risk-adjusted
  basis over the sample.
- VW(-50%) and VW(NZ) deliver nearly identical financial outcomes:
  return 7.79% vs 7.70%, Sharpe 0.444 vs 0.437, TE 3.24% vs 3.16%.

## Does net-zero reduce carbon more strongly than the 50% strategy?

- Yes, from ~2017 onwards. The NZ cap path
  `(1 - 0.10)^(Y - 2013 + 1) * CF(P_vw)_2013` decays from 204.4 (allocation
  year 2013) to 64.1 (allocation year 2024), see
  `part4_netzero_target_vs_realized_cf.csv` and
  `part4_cf_target_vs_realized.png`.
- In 2013-2015 the cap is loose enough that the TE-optimal portfolio is
  already below it (slack 5 - 21 tCO2e/M$), so VW(NZ) and VW(-50%) coincide.
- From 2017 onwards the cap binds every year (`slack` collapses to
  numerical zero) and VW(NZ) lies *below* 50% of CF(P_vw): in 2024 the cap
  is 64.1 vs 0.5 * CF(P_vw)=73.9.

## Financial cost of net-zero

- ~0.09 pp annualised return below VW(-50%) over 2014-2025 (7.70% vs 7.79%);
  ~0.007 lower Sharpe. Negligible at this sample size.
- The financial cost vs VW (the benchmark) is *negative* in this sample:
  the dynamic decarbonisation path *adds* +0.98 pp annualised return.

## Tracking-error cost

- Ex-post TE = 3.16% (slightly *below* VW(-50%)). Ex-ante TE
  (`part4_tracking_error_vw_netzero.png`) rises from ~2.5% in 2013 to ~4.5%
  in 2024 as the cap tightens, as expected.

## Concentration / tilt

- `outputs/tables/part4_weights_vw_netzero.csv`: the portfolio holds the bulk
  of the eligible universe (200-400 names per year) but progressively cuts
  the top CF-per-weight names (mining majors, integrated steel, utilities).
- Tilt is *cumulative*: the same names that were trimmed in 2017 are
  excluded entirely by 2022-2024.

## Feasibility of the dynamic constraint

- All 12 years feasible. Min slack across the path -1.4e-14 (numerical
  noise). Optimiser succeeded every year.
- The cap stays achievable because the Pacific universe still contains a
  large mass of low-CF firms (services, tech, healthcare) whose weight the
  TE objective can shift to. No hidden infeasibility had to be relaxed.

## Consistency with course logic

- The strategy follows the prescribed decarbonisation path: anchor at
  CF(P_vw)_2013, tighten 10% per year, no re-anchoring.
- It illustrates the standard PAII/PAB story: a passive investor can
  follow a 10%-per-year carbon trajectory by progressively tilting away
  from the CF tail at a small TE cost, with no observed return penalty on
  this sample.
- The fact that the constraint is not binding in 2013-2015 is a feature
  of this universe (already low-CF firms available); it shows the green
  budget is loose at the start of the path and would only become
  meaningfully restrictive if the universe shifted or theta increased.

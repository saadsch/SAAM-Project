# Validation Checklist — Parts 3 and 4

This checklist documents the automated validation block included in
`src/SAAM_Project_FINAL.ipynb` (Section 8). All checks are recomputed
from the CSVs in `outputs/tables/` on every notebook run; this file
records what is being checked and the acceptance criteria.

## 1. Weight feasibility (long-only, fully invested)

For every rebalance year and every portfolio
(`P_mv`, `P_vw`, `P_mv(0.5)`, `P_vw(0.5)`, `P_vw(NZ)`):

- **Sum of weights**: `|sum(w) - 1| < 1e-6`
- **No short positions**: `min(w) >= -1e-9`
- **No NaN**: `w.notna().all()`

Source files:
- `outputs/tables/mv_weights.csv`
- `outputs/tables/part3_weights_mv_carbon50.csv`
- `outputs/tables/part3_weights_vw_carbon50.csv`
- `outputs/tables/part4_weights_vw_netzero.csv`

## 2. Constraint slack near zero (carbon constraint binding)

The Part-3 and Part-4 SLSQP solutions should sit exactly on the
carbon-budget constraint. We verify:

- `max(|slack|) < 1e-6` for `part3_constraint_slack_mv_carbon50.csv`
- `max(|slack|) < 1e-6` for `part3_constraint_slack_vw_carbon50.csv`
- `max(|slack|) < 1e-6` for `part4_constraint_slack_vw_netzero.csv`

Observed: all three at numerical noise (max abs slack ≈ 1.7e-13).

## 3. 50% carbon reduction (Part 3)

For each rebalance year:
- `CF(P_vw(0.5))_Y <= 0.5 * CF(P_vw)_Y * (1 + 1e-6)`
- `CF(P_mv(0.5))_Y <= 0.5 * CF(P_mv)_Y * (1 + 1e-6)` (when applicable)

Source: `outputs/tables/part3_summary_*.csv` and
`outputs/tables/part3_carbon_metrics_mv_vw.csv`.

## 4. Net-zero trajectory (Part 4)

The realized footprint of `P_vw(NZ)` must respect the rolling cap
`cap_Y = 0.9^(Y - 2013 + 1) * CF(P_vw)_2013`:

- `realized_cf_Y <= cap_Y * (1 + 1e-6)` for every year `Y` in 2013..2024
- Anchor: `cap_2013 = 0.9 * CF(P_vw)_2013` (one-step decay from t=0)
- Anchor numeric: `CF(P_vw)_2013 = 227.10` tCO2e per USD-million invested

Source: `outputs/tables/part4_netzero_target_vs_realized_cf.csv`.

## 5. Cumulative paths start at one

For every portfolio's monthly return series:
- `(1 + r).cumprod().iloc[0]` is finite
- The OOS cumulative growth path starts strictly above zero
- Sample period: 2014-01-31 through 2025-12-31, exactly 144 months

Source files:
- `outputs/tables/part3_returns_mv_carbon50.csv`
- `outputs/tables/part3_returns_vw_carbon50.csv`
- `outputs/tables/part4_returns_vw_netzero.csv`

## 6. Tracking error — independent recomputation

For `P_vw(0.5)` and `P_vw(NZ)`:

- **Ex-post TE** = `std(r_p - r_vw) * sqrt(12)` recomputed from the
  monthly returns CSVs must match the values in
  `part3_tracking_error_vw_carbon50.csv` / `part4_tracking_error_vw_netzero.csv`
  to within 1e-8.
- **Ex-ante TE** = `sqrt((w_p - w_vw)' Σ (w_p - w_vw) * 12)` is small
  (≈ 0) by construction of the in-sample optimization. This is **not** a
  bug — see Section 6.4 of the notebook for a discussion of sample
  covariance rank-deficiency (N ≈ 260–470 firms vs. 120 monthly obs)
  and why the OOS ex-post TE re-emerges at a higher level.

## Acceptance

The notebook prints a green "ALL CHECKS PASSED" line at the end of
Section 8 if and only if all of the above hold. If any check fails the
cell raises an `AssertionError` with the offending portfolio/year.

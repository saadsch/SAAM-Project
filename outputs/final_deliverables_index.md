# Final Deliverables Index — SAAM Pacific Carbon-Aware Allocation

**Course**: SAAM — Sustainability-Aware Asset Management (HEC Lausanne, Prof. E. Jondeau)
**Region**: Pacific
**Implementation window**: 2014-01 to 2025-12 (144 monthly returns)
**Estimation window**: 120 months, rebalance at Dec of year `Y`, hold for year `Y+1`

## Primary notebook

| File | Purpose |
| --- | --- |
| `src/SAAM_Project_FINAL.ipynb` | Single submission notebook: Parts I, II, III and IV in one runnable document, with markdown explanations, validation, limitations and LLM disclosure. |

## Supporting notebooks (reference / construction)

| File | Purpose |
| --- | --- |
| `src/data_cleaning.ipynb` | Section 2.1 cleaning rules: price < 0.5 → NaN, ffill middle gaps, trailing → 0, returns. |

The final, submission-relevant numbers are those in `SAAM_Project_FINAL.ipynb`,
which restricts every part (including Part I and Part II) to the carbon-eligible
universe so that all four parts share the same firm set and the same 144-month
out-of-sample window. Standalone Part-1 (`MVP-construction.ipynb`) and Part-2
(`vwp.ipynb`) notebooks that ran on the broader returns-eligible universe were
removed (see `cleanup_log.md`) because their summary statistics did not match
those of the final pipeline.

## Core Python modules

| File | Purpose |
| --- | --- |
| `src/saam_core.py` | Optimizers (long-only MV, TE-min with carbon constraint), drift simulator, summary statistics (geometric annualized return, ex-post TE, Sharpe on monthly excess). |
| `src/04_part2_vw.py` | Part 2 driver (cap-weighted, monthly rebalance via `cap.shift(1)`). |
| `src/05_part3.py` | Part 3 driver (50% carbon reduction). |
| `src/06_part4_netzero.py` | Part 4 driver (net-zero, θ=0.10, anchor 2013). |

## Output tables — `outputs/tables/`

| File | Content |
| --- | --- |
| `part3_weights_mv_carbon50.csv` | Part 1 and Part 3.2 weights at each Dec-Y rebalance (`portfolio ∈ {mv, mv_50}`). The `mv` rows are the unconstrained MV weights consumed by Part I; the `mv_50` rows are the 50%-carbon-constrained MV weights consumed by Part 3.2. |
| `part3_weights_vw_carbon50.csv` | Part 2 and Part 3.3 weights at each Dec-Y rebalance (`portfolio ∈ {vw, vw_50}`). |
| `part3_carbon_metrics_mv_vw.csv` | WACI and CF for `P_mv` and `P_vw` each year. |
| `part3_top10_waci_contributors.csv` | Top-10 firms by `w_i * CI_i`. |
| `part3_top10_cf_contributors.csv` | Top-10 firms by `w_i * E_i / Cap_i`. |
| `part3_returns_mv_carbon50.csv` | Monthly returns of `P_mv` and `P_mv(0.5)`. |
| `part3_returns_vw_carbon50.csv` | Monthly returns of `P_vw(0.5)`. |
| `part3_constraint_slack_mv_carbon50.csv` | Carbon constraint slack (≈ 0) per year. |
| `part3_constraint_slack_vw_carbon50.csv` | Carbon constraint slack (≈ 0) per year. |
| `part3_summary_mv_vs_mv_carbon50.csv` | Performance summary: `P_mv` vs `P_mv(0.5)`. |
| `part3_summary_vw_vs_vw_carbon50.csv` | Performance summary: `P_vw` vs `P_vw(0.5)`. |
| `part3_tracking_error_vw_carbon50.csv` | Ex-ante and ex-post TE of `P_vw(0.5)` vs `P_vw`. |
| `part4_weights_vw_netzero.csv` | Part-4 net-zero weights. |
| `part4_returns_vw_netzero.csv` | Monthly returns of `P_vw(NZ)`. |
| `part4_constraint_slack_vw_netzero.csv` | NZ constraint slack (≈ 0) per year. |
| `part4_summary_vw_vs_carbon50_vs_netzero.csv` | Performance: `P_vw`, `P_vw(0.5)`, `P_vw(NZ)`. |
| `part4_tracking_error_vw_netzero.csv` | Ex-ante and ex-post TE of `P_vw(NZ)` vs `P_vw`. |
| `part4_netzero_target_vs_realized_cf.csv` | Cap trajectory vs realized CF per year. |

## Output figures — `outputs/figures/`

Part 1:
- `part1_cumulative_mv.png` — generated inside `SAAM_Project_FINAL.ipynb`.

Part 2:
- `part2_cumulative_vw.png` — generated inside `SAAM_Project_FINAL.ipynb`.
- `part2_cumulative_mv_vs_vw.png` — generated inside `SAAM_Project_FINAL.ipynb`.

Part 3:
- `part3_waci_mv_vs_vw.png` — WACI per year, MV vs VW.
- `part3_carbon_footprint_mv_vs_vw.png` — CF per year, MV vs VW.
- `part3_waci_mv_vs_mv_carbon50.png` — WACI: `P_mv` vs `P_mv(0.5)`.
- `part3_cf_mv_vs_mv_carbon50.png` — CF: `P_mv` vs `P_mv(0.5)`.
- `part3_waci_vw_vs_vw_carbon50.png` — WACI: `P_vw` vs `P_vw(0.5)`.
- `part3_cf_vw_vs_vw_carbon50.png` — CF: `P_vw` vs `P_vw(0.5)`.
- `part3_cumulative_mv_vs_mv_carbon50.png` — cumulative path: `P_mv` vs `P_mv(0.5)`.
- `part3_cumulative_vw_vs_vw_carbon50.png` — cumulative path: `P_vw` vs `P_vw(0.5)`.
- `part3_tracking_error_vw_carbon50.png` — ex-ante vs ex-post TE.

Part 4:
- `part4_cf_target_vs_realized.png` — net-zero cap vs realized CF.
- `part4_waci_vw_vs_carbon50_vs_netzero.png` — WACI for the three VW portfolios.
- `part4_cumulative_vw_vs_carbon50_vs_netzero.png` — cumulative paths.
- `part4_tracking_error_vw_netzero.png` — ex-ante vs ex-post TE.

## Documentation

| File | Purpose |
| --- | --- |
| `outputs/cleanup_log.md` | Stale files removed and why. |
| `outputs/validation_checklist_part3_part4.md` | What the in-notebook validation cell checks. |
| `outputs/final_deliverables_index.md` | This file. |
| `README.md` | Repository overview. |
| `requirements.txt` | Python dependencies. |

## Key headline numbers (loaded from CSVs, do not hard-code)

(For convenience — these are the values printed by the final notebook.)

| Portfolio | Ann. ret | Vol | Sharpe (excess) | Cum. ret | Ex-post TE vs VW |
| --- | ---: | ---: | ---: | ---: | ---: |
| `P_mv`        | 8.30% | 11.58% | 0.565 | 160.32% | — |
| `P_vw`        | 6.72% | 12.88% | 0.386 | 118.34% | — |
| `P_mv(0.5)`   | 9.06% | 11.70% | 0.624 | 183.14% | — |
| `P_vw(0.5)`   | 7.79% | 12.86% | 0.444 | 146.02% | 3.24% |
| `P_vw(NZ)`    | 7.70% | 12.89% | 0.437 | 143.65% | 3.16% |

Anchor: `CF(P_vw)_2013 = 227.10` tCO2e per USD-million invested.

## How to reproduce

From the repository root:

```
python3 -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 src/SAAM_Project_FINAL.ipynb
```

This regenerates every figure, table and summary above from
`data/processed/`. No notebook cell relies on a hard-coded result.

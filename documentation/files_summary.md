# Summary of the new files

This document explains exactly what each new file does, what techniques it
uses, and what every important variable, constant and helper function
represents. It is intended as a reading guide for the final report.

---

## 1. Optimisation engine: `src/saam_core.py`, `src/05_part3.py`, `src/06_part4.py`

### Purpose

`src/saam_core.py` is a shared helper module: it loads the panels, applies
the PDF Section 2.1 price-cleaning rules, builds the eligible universe at
end of each allocation year, solves the long-only QP with SLSQP, simulates
the drift-based implementation year, computes the monthly-rebalanced VW
benchmark and the headline statistics.

`src/05_part3.py` runs Part III — portfolios `mv` (PDF Section 2.2),
`mv_50` (Section 3.2) and `vw_50` (Section 3.3) — with explicit
`# === Section X.Y === #` markers in the code.

`src/06_part4.py` runs Part IV — portfolio `vw_nz` (Section 4) — anchored
on `CF(P_vw)_{2013}` with cap path `(1 - θ)^{Y - 2013 + 1} · CF(P_vw)_{2013}`
(`θ = 10 %`).

Both scripts produce `vw` (monthly rebalanced) and `vw_drift` (annual-cap,
drift). Allocation years 2013–2024, performance 2014-01 → 2025-12.

### Technique

* **Solver.** `scipy.optimize.minimize` with the **SLSQP** method
  (Sequential Least-SQuares Programming). SLSQP is a standard SQP solver
  for convex QPs; it accepts a quadratic objective, linear equality and
  inequality constraints, and box bounds directly. We pass closed-form
  analytic gradients for both the variance and the tracking-error
  objectives, which makes convergence fast and deterministic.

* **Covariance estimation.** For each year `Y` we use the trailing 120
  monthly returns. We compute `Σ_Y` on **complete-case rows** (rows of
  the panel with no missing value across the eligible firms). If fewer
  than 36 complete rows are available we fall back to filling each
  column with its in-window mean. A ridge `1e-6 · I` is added to `Σ_Y`
  before the optimisation for numerical stability (this guarantees a
  well-defined Cholesky factor even when several firms have very
  collinear returns).

* **Initialisation of the optimiser.**
  - For min-variance (`mv`, `mv_50`) we start at `1/n` (the uniform
    point on the simplex).
  - For tracking-error (`vw_50`, `vw_nz`) we start at the benchmark
    `w_vw`. If the benchmark violates the carbon cap, we mix it with
    the lowest-carbon vertex of the simplex (the indicator on the
    asset with the smallest `E_i/Cap_i`) using the smallest mixing
    coefficient that puts the start point inside the cap. This is the
    fix that makes the SLSQP solution sit *on* the carbon boundary,
    which is where the constrained optimum lies whenever the
    unconstrained optimum (`w_vw`) violates the cap.

* **Performance simulation.** PDF Section 2.2 drift formula:
  `w_{i,t+k} = w_{i,t+k-1} · (1 + R_{i,t+k}) / (1 + R_{p,t+k})`,
  with `R_{p,t+k} = w_{t+k-1}^⊤ R_{t+k}`. No rebalancing inside the
  implementation year. Delisting (1 + R_p ≤ 0) zeroes the portfolio.

* **VW benchmark (strict PDF Section 2.3).** Computed with **monthly
  rebalancing** using the end-of-month capitalisation file
  `DS_MV_T_USD_M.csv`: `R^{vw}_{t+1} = Σ_i (Cap_{i,t} / Σ Cap_{j,t}) R_{i,t+1}`.
  An additional drift-based annual variant `vw_drift` is also produced
  and used as the *benchmark vector* inside the tracking-error problems
  (which require an annual α^{(vw)}_Y).

### Key constants

| Name | Meaning |
|---|---|
| `YEAR_FIRST = 2013`, `YEAR_LAST = 2024` | Allocation years (end-of-year decisions). |
| `PERF_START = 2014`, `PERF_END = 2025` | Implementation years (monthly performance window). |
| `ESTIM_MONTHS = 120` | 10 years of monthly returns used to estimate `μ_Y` and `Σ_Y`. |
| `MIN_OBS = 36` | Minimum observed monthly returns per firm in the window. |
| `STALE_THRESHOLD = 0.50` | Maximum share of zero monthly returns allowed in the window. |
| `STARTING_WEALTH_MUSD = 1.0` | V_{2013} = USD 1 million (used to attribute portfolio emissions). |
| `CARBON_SCOPE = "scope1"` | Strategy assigned to the group; switch to `"scope2"` or `"scope1_plus_scope2"` as needed. |
| `THETA_NZ = 0.10` | 10 % per-year carbon-footprint reduction in the net-zero path. |
| `RIDGE_EPS = 1e-6` | Diagonal regulariser added to `Σ_Y`. |

### Key per-year objects (built inside `_build_year_inputs`)

The function returns a `YearInputs` dataclass with everything the optimiser
needs for one year:

| Field | Meaning |
|---|---|
| `year` | Allocation year `Y`. |
| `isins` | Sorted list of eligible ISINs at end of `Y`. |
| `names`, `countries` | Aligned company name and country (Pacific). |
| `rets_est` | Monthly returns used to estimate `Σ_Y` (last 120 months, eligible firms). |
| `rets_oos` | Monthly returns of the next 12 months (used for simulation). |
| `mu` | Mean monthly return per asset over the estimation window. |
| `cov` | Estimated covariance matrix `Σ_Y` (after complete-case estimation + ridge). |
| `cap_y` | End-of-year market capitalisation per ISIN. |
| `emissions` | Annual emissions `E_{i,Y}` per ISIN (tCO₂e). |
| `revenues` | Annual revenue `Rev_{i,Y}` (in thousands USD). |
| `ci` | Carbon intensity `CI_i = E_i / (Rev_i / 1000)` (tCO₂e per USD-million revenue). |
| `cf_per_w` | `E_i / Cap_i` — the contribution to the portfolio carbon footprint per unit weight. |
| `w_vw` | End-of-year value weights `Cap_i / Σ Cap_j`. |

### Other named quantities

| Name | Meaning |
|---|---|
| `cf_vw_2013` | Carbon footprint of the VW portfolio at end of 2013. Used to anchor the net-zero path. |
| `nz_limit` | Net-zero carbon cap for year `Y`: `(1 − θ)^{Y − YEAR_FIRST + 1} · cf_vw_2013`. |
| `cf_mv` | Carbon footprint of `mv` for year `Y`. Used as the reference for `mv_50` (`cap = 0.5 · cf_mv`). |
| `cf_vw` | Carbon footprint of `vw` for year `Y`. Used as the reference for `vw_50` (`cap = 0.5 · cf_vw`). |

### Optimisation problems solved (recap)

| Portfolio | Objective | Inequality constraint |
|---|---|---|
| `mv`    | min `wᵀΣw`            | — |
| `mv_50` | min `wᵀΣw`            | `cf_per_w · w ≤ 0.5 · cf_mv` |
| `vw_50` | min `(w − w_vw)ᵀΣ(w − w_vw)` | `cf_per_w · w ≤ 0.5 · cf_vw` |
| `vw_nz` | min `(w − w_vw)ᵀΣ(w − w_vw)` | `cf_per_w · w ≤ nz_limit` |

All four share `Σ w = 1` and `0 ≤ w_i ≤ 1`.

### Outputs (under `outputs/`)

**From `src/05_part3.py` (Part III)**

| File | Content |
|---|---|
| `part3_monthly_returns.csv` | Monthly returns 2014-01 to 2025-12 for `vw`, `vw_drift`, `mv`, `mv_50`, `vw_50`. |
| `part3_performance_summary.csv` | Annualised return, vol, RF, Sharpe (excess-return with Pacific RF from `rf_rate.csv`), TE vs `vw`, min/max monthly return, cumulative return. |
| `part3_annual_carbon_metrics.csv` | Per year and per Part III portfolio: WACI, CF, attributed emissions, cap, optimisation status. |
| `part3_portfolio_weights.csv` | Non-zero weights per year, per portfolio, per ISIN (with name, country, CI). |
| `part3_top10_waci_drivers_by_year.csv` | Section 3.1 — top-10 firms ranked by `w_vw · CI` per year. |
| `part3_top10_cf_drivers_by_year.csv` | Section 3.1 — top-10 firms ranked by `w_vw · E/Cap` per year. |
| `part3_tracking_error.csv` | Annualised ex-ante and ex-post TE per year × portfolio. |
| `part3_optimizer_status.csv` | SLSQP success/message per year × portfolio. |
| `part3_carbon_constraint_slack.csv` | `carbon_limit − realised_CF` with a `binding_within_1e-6` flag. |
| `part3_exclusions_overweights.csv` | Section 3.4 — top-10 exclusions and overweights vs VW per year × portfolio. |
| `figures/part3_1_waci.png` | Section 3.1 — annual WACI by portfolio. |
| `figures/part3_1_carbon_footprint.png` | Section 3.1 — annual CF by portfolio. |
| `figures/part3_2_mv_vs_mv50_cumulative.png` | Section 3.2 — cumulative growth-of-$1: VW, MV, MV(0.5). |
| `figures/part3_3_vw_vs_vw50_cumulative.png` | Section 3.3 — cumulative growth-of-$1: VW vs VW(0.5). |
| `figures/part3_4_tracking_error.png` | Section 3.4 — annual ex-ante TE per constrained portfolio. |

**From `src/06_part4.py` (Part IV)**

| File | Content |
|---|---|
| `part4_monthly_returns.csv` | Monthly returns 2014-01 to 2025-12 for `vw`, `vw_drift`, `vw_nz`. |
| `part4_performance_summary.csv` | RF-adjusted Sharpe + TE vs `vw` for the Part IV portfolios. |
| `part4_annual_carbon_metrics.csv` | Per year for `vw_drift` and `vw_nz`. |
| `part4_portfolio_weights.csv` | Non-zero weights of `vw_nz` (and `vw_drift`) per year × ISIN. |
| `part4_netzero_path.csv` | Section 4.1 — anchor `CF(P_vw)_{2013}`, cap target, realised `CF(vw_nz)` and `CF(vw)` per year. |
| `part4_tracking_error.csv` | Annualised ex-ante and ex-post TE. |
| `part4_optimizer_status.csv` | SLSQP success/message. |
| `part4_carbon_constraint_slack.csv` | Cap vs realised CF with binding flag. |
| `part4_exclusions_overweights.csv` | Top-10 exclusions and overweights of `vw_nz` vs VW. |
| `figures/part4_1_netzero_cumulative.png` | Section 4 — cumulative growth-of-$1: VW vs VW(NZ). |
| `figures/part4_2_netzero_path.png` | Section 4 — target NZ cap path vs realised CF of `vw_nz` and `vw`. |

---

## 2. `src/Part3_Part4_notebook.ipynb` — reproducible notebook

### Purpose

Single-notebook deliverable required by PDF Section 5.2: when run top to
bottom, it reproduces every table and figure of the report. No hard-coded
paths, no hidden setup.

### Technique

The notebook is **a thin wrapper around the two engines** (`05_part3.py`
and `06_part4.py`), so the scripts remain the single source of truth. The
notebook imports each script with `importlib.util.spec_from_file_location`,
calls its `run()` function, then reloads the resulting CSVs/PNGs from
`outputs/` to render them. The performance, carbon-metric and weight
tables in the notebook concatenate the Part III outputs with the `vw_nz`
rows from the Part IV outputs.

### Cells (10 sections)

1. **Setup.** Locate the project root regardless of where the notebook is
   opened from (project root or `src/`); resolve `OUTPUTS` and `FIGURES`
   paths; make `src/` importable so `saam_core` is reachable.
2. **Run engines.** Imports and executes `run()` for both `05_part3.py`
   and `06_part4.py`.
3. **Performance summary.** Reads `part3_performance_summary.csv` plus
   the `vw_nz` row of `part4_performance_summary.csv`.
4. **Annual carbon metrics + constraint check.** Reads
   `part3_annual_carbon_metrics.csv` plus the `vw_nz` rows of
   `part4_annual_carbon_metrics.csv`; computes the maximum excess of
   `CF` above its `carbon_limit` per portfolio (should be ≈ 0).
5. **Top-10 WACI drivers.** Reads `part3_top10_waci_drivers_by_year.csv`
   (universe-level WACI drivers do not depend on the portfolio).
6. **Exclusions / overweights vs VW.** Concatenates
   `part3_exclusions_overweights.csv` with the `vw_nz` rows of
   `part4_exclusions_overweights.csv`; shows the firms excluded by
   `vw_nz` and the firms most overweighted vs VW in the latest year.
7. **Country tilts.** Aggregates the weights file by country to display
   the geographic distribution per portfolio per year — the empirical
   answer to the "sector/country tilts" discussion the PDF asks for.
8. **Figures.** Displays the six PNGs inline.
9. **Interpretation.** Markdown paragraph contrasting the five
   portfolios — financial vs carbon trade-off, MV's structural carbon
   excess, NZ tightening, etc.
10. **Limitations and mandatory LLM disclosure.** The disclosure
    paragraph satisfies the PDF requirement that every report must
    include a labelled "Use of Large Language Models (LLMs)" paragraph.

---

## 3. `documentation/part3_4_methodology.md` — methodology note

### Purpose

Written explanation for the final report. Designed to be quoted directly
in the methodology section, with no LLM-style padding.

### Sections

1. **Project refresher.** Restates the five portfolios and the CF/WACI
   formulas.
2. **File map.** Lists the files in `src/` and `documentation/` and
   says, in one sentence, what each one does.
3. **Methodology of `05_part3.py` / `06_part4.py` (shared `saam_core.py`).** SLSQP, benchmark warm-start,
   complete-case `Σ`, the strict monthly-rebalanced VW benchmark, and
   the two new diagnostic outputs (exclusions/overweights and the
   net-zero path plot).
4. **Headline results.** Table of annualised return, volatility and
   Sharpe per portfolio (Scope 1, Pacific, 2014–2025), plus
   confirmation that the carbon caps bind tightly.
5. **Interpretation prompts.** Specific points to make in the report
   about each portfolio.
6. **Limitations.** Data coverage, estimation error, Scope 1 vs
   Scope 1+2, sector concentration, NZ base-year sensitivity.
7. **LLM disclosure.** Mandatory paragraph per the PDF.

---

## 4. This file — `documentation/files_summary.md`

A reading guide that maps each new file to its purpose, techniques,
variables, and outputs. Use it to navigate the deliverable.

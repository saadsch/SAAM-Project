# Summary of the new files

This document explains exactly what each new file does, what techniques it
uses, and what every important variable, constant and helper function
represents. It is intended as a reading guide for the final report.

---

## 1. `src/05_part3_part4.py` — optimisation engine

### Purpose

Solves Parts III & IV (portfolios `mv`, `mv_50`, `vw_50`, `vw_nz`) and
produces the value-weighted benchmark (`vw` and `vw_drift`) year by year
from 2013 to 2024 (allocation years), with monthly performance simulated
from 2014 to 2025.

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

| File | Content |
|---|---|
| `part3_4_monthly_portfolio_returns.csv` | Monthly returns 2014-01 to 2025-12 for `vw`, `vw_drift`, `mv`, `mv_50`, `vw_50`, `vw_nz`. |
| `part3_4_performance_summary.csv` | Annualised return, annualised volatility, Sharpe (with 0% RF), min/max monthly return, cumulative return. |
| `part3_4_annual_carbon_metrics.csv` | Per year and per portfolio: WACI, carbon footprint, attributed emissions, the applicable cap, optimisation status. |
| `part3_4_portfolio_weights.csv` | Non-zero weights per year, per portfolio, per ISIN (with name, country, CI). |
| `part3_4_top10_waci_drivers_by_year.csv` | Top-10 firms ranked by `w_vw · CI` per year. |
| `part3_4_exclusions_overweights.csv` | Per year and per constrained portfolio: top-10 exclusions vs VW (firms held by VW but excluded) and top-10 overweights vs VW. |
| `figures/part3_mv_vs_mv50_cumulative.png` | Cumulative growth of $1: VW (monthly), MV, MV(0.5). |
| `figures/part3_vw_vs_vw50_cumulative.png` | Cumulative growth of $1: VW vs VW(0.5). |
| `figures/part4_vw_vw50_netzero_cumulative.png` | Cumulative growth of $1: VW, VW(0.5), VW(NZ). |
| `figures/part3_4_waci_by_portfolio.png` | Annual WACI for every portfolio. |
| `figures/part3_4_carbon_footprint_by_portfolio.png` | Annual carbon footprint for every portfolio. |
| `figures/part4_netzero_path.png` | Net-zero target path overlaid on realised CF of `vw_drift` and `vw_nz`. |

---

## 2. `src/Part3_Part4_notebook.ipynb` — reproducible notebook

### Purpose

Single-notebook deliverable required by PDF Section 5.2: when run top to
bottom, it reproduces every table and figure of the report. No hard-coded
paths, no hidden setup.

### Technique

The notebook is **a thin wrapper around the engine** (`05_part3_part4.py`),
so the script remains the single source of truth. We import the script
with `importlib.util.spec_from_file_location`, then call `part3_4.run()`,
which writes all CSVs and PNGs into `outputs/`. The notebook reloads
those artefacts and renders them.

### Cells (10 sections)

1. **Setup.** Locate the project root regardless of where the notebook is
   opened from (project root or `src/`); resolve `OUTPUTS` and `FIGURES`
   paths; ensure the folders exist.
2. **Run engine.** Imports and executes `part3_4.run()`.
3. **Performance summary.** Reads `part3_4_performance_summary.csv` and
   renders the headline table.
4. **Annual carbon metrics + constraint check.** Reads
   `part3_4_annual_carbon_metrics.csv`; computes the maximum excess of
   `CF` above its `carbon_limit` per portfolio (should be ≈ 0).
5. **Top-10 WACI drivers.** Reads
   `part3_4_top10_waci_drivers_by_year.csv` (used to answer the PDF's
   Section 3.1 question "which firms drive the WACI up").
6. **Exclusions / overweights vs VW.** Reads
   `part3_4_exclusions_overweights.csv` and shows, for the latest
   allocation year, the firms excluded by `vw_nz` and the firms most
   overweighted vs VW.
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
3. **Methodology of `05_part3_part4.py`.** SLSQP, benchmark warm-start,
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

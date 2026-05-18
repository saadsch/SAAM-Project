# Handover note — Parts III & IV

Everything a teammate needs to read in order to take over Parts III and IV
of the SAAM 2026 project. Read this file first; the other docs go deeper
on specific topics.

---

## 1. What the project asks for

Group strategy: **Pacific region, Scope 1 emissions, 50 % reduction target,
10 % per-year net-zero path.** Construction window 2013–2024 (annual
rebalancing), implementation window 2014–2025 (monthly).

Five portfolios are constructed:

| Code | Description | Optimisation problem |
|---|---|---|
| `vw` | Value-weighted benchmark (monthly rebalanced, strict PDF Section 2.3) | weights = `Cap_{i,t} / Σ_j Cap_{j,t}` |
| `vw_drift` | Value-weighted at end of each year, drifts within the year (= benchmark vector for tracking-error problems) | `w = Cap_{i,Y} / Σ_j Cap_{j,Y}` |
| `mv` | Long-only minimum variance | `min wᵀΣw` s.t. `Σw=1`, `w≥0` |
| `mv_50` | Min variance with 50 % carbon cut vs `mv` (PDF 3.2) | `min wᵀΣw` s.t. `Σw=1`, `w≥0`, `c·w ≤ 0.5·CF(mv)` |
| `vw_50` | Tracking error vs `vw_drift` with 50 % carbon cut vs `vw` (PDF 3.3) | `min (w-w_vw)ᵀΣ(w-w_vw)` s.t. `Σw=1`, `w≥0`, `c·w ≤ 0.5·CF(vw)` |
| `vw_nz` | Tracking error with 10 %-per-year carbon path (PDF 4.1) | same as `vw_50` but cap = `(0.9)^(Y-2012)·CF(vw)_{2013}` |

Carbon definitions:
* `CI_i = E_i / (Rev_i / 1000)` — carbon intensity, tCO₂e per USD-million revenue.
* `e_i = E_i / Cap_i` — CF contribution per unit weight, tCO₂e per USD-million invested.
* Portfolio CF: `CF(w) = c · w` where `c = (e_1, …, e_n)`.
* Portfolio WACI: `WACI(w) = w · CI`.

---

## 2. Files (start here)

```
src/01_cleaning.py            # Datastream → Pacific universe (Part I prerequisite)
src/02_analysis.py            # Price cleaning, universe filtering (Part I prerequisite)
src/03_outputs.py             # Part I outputs (risk vs CI scatter)
src/saam_core.py              # Shared helpers (loading, cleaning, SLSQP, stats)
src/05_part3.py               # Part III optimisation engine (Sections 3.1-3.4)
src/06_part4.py               # Part IV optimisation engine (Section 4, Net-Zero)
src/Part3_Part4_notebook.ipynb  # Reproducible notebook deliverable  ← RUN THIS
documentation/SAAM_Project_2026 (1).pdf
documentation/handover.md       # this file
documentation/files_summary.md  # file-by-file reading guide (longer)
documentation/part3_4_methodology.md  # methodology note for the report
documentation/part1_methodology.md
```

To regenerate every output from scratch:

```bash
cd /Users/yazidabaroudi/SAAM-Project-1
python3 src/01_cleaning.py        # only if data/processed/ is empty
python3 src/02_analysis.py        # only if data/processed/ is empty
python3 src/05_part3.py
python3 src/06_part4.py
```

Or open `src/Part3_Part4_notebook.ipynb` in VS Code and run all cells.

---

## 3. Technique used in `src/05_part3.py` and `src/06_part4.py`

Both scripts share `src/saam_core.py`. Part III handles `mv`, `mv_50`, `vw_50`; Part IV handles `vw_nz`.

* **Solver:** `scipy.optimize.minimize(method="SLSQP")` with analytic
  gradients. SLSQP handles a quadratic objective + linear equality and
  inequality constraints + box bounds natively. Standard, transparent,
  fast enough for ≤500 firms.
* **Why not a hand-rolled projected-gradient method?** Earlier ChatGPT
  code used one. Its early-stop test fired on the second iteration
  every year, leaving the optima under-converged. Headline Sharpes
  jumped from 0.60 → 0.72 (`mv`) and 0.56 → 0.77 (`mv_50`) after the
  switch to SLSQP.
* **Covariance estimation:** complete-case rows of the trailing 120-month
  window (fallback to column-mean fill if <36 complete rows), plus a
  ridge `1e-6·I` so Σ is strictly positive definite.
* **Optimiser initial point:**
  * `1/n` for the variance objective;
  * benchmark `w_vw` for the tracking-error objective, mixed with the
    lowest-carbon vertex of the simplex if `w_vw` violates the cap. This
    is what makes the constrained optimum sit on the carbon boundary
    (which is where it lies whenever the unconstrained optimum is
    infeasible).
* **Performance simulation:** drift formula `w_{t+1} = w_t·(1+R)/(1+R_p)`
  (PDF 2.2), no rebalancing inside the year, delisting wipes the portfolio.
* **VW benchmark:** computed two ways. `vw` uses **monthly** caps with
  monthly rebalancing (strict PDF 2.3). `vw_drift` uses end-of-year caps
  with drift — this is the benchmark vector that enters the tracking-error
  problems.

---

## 4. Variables to know

### Constants at the top of `src/saam_core.py`

| Constant | Value | Meaning |
|---|---|---|
| `YEAR_FIRST` | 2013 | first allocation year |
| `YEAR_LAST` | 2024 | last allocation year |
| `PERF_START` | 2014 | first implementation month |
| `PERF_END` | 2025 | last implementation month |
| `ESTIM_MONTHS` | 120 | length of the rolling window |
| `MIN_OBS` | 36 | minimum monthly observations to be eligible |
| `STALE_THRESHOLD` | 0.50 | max share of zero monthly returns |
| `STARTING_WEALTH_MUSD` | 1.0 | V_{2013} = USD 1 million |
| `CARBON_SCOPE` | "scope1" | switch to `"scope2"` or `"scope1_plus_scope2"` if instructed |
| `THETA_NZ` | 0.10 | 10 % annual carbon reduction in NZ |
| `RIDGE_EPS` | 1e-6 | Σ regularisation |

### Per-year object `YearInputs` (built in `_build_year_inputs`)

| Field | Meaning |
|---|---|
| `year` | allocation year `Y` |
| `isins` | eligible ISINs at end of `Y` |
| `names`, `countries` | aligned identification |
| `rets_est` | 120-month estimation window (eligible firms) |
| `rets_oos` | next 12 months' returns (simulation) |
| `mu` | mean monthly return per firm |
| `cov` | Σ_Y after ridge |
| `cap_y` | end-of-year market cap per firm |
| `emissions` | annual emissions per firm |
| `revenues` | annual revenue per firm (kUSD) |
| `ci` | `CI_i = E_i / (Rev_i / 1000)` |
| `cf_per_w` | `e_i = E_i / Cap_i` |
| `w_vw` | annual value-weights `Cap_i / Σ Cap_j` |

### Scalars in the main loop

| Variable | Definition |
|---|---|
| `cf_mv` | `CF` of the `mv` portfolio in year `Y` |
| `cf_vw` | `CF` of `vw_drift` in year `Y` |
| `cf_vw_2013` | `CF(vw)` for `Y = 2013`, anchor of the NZ trajectory |
| `nz_limit` | `(1-θ)^(Y-2012)·cf_vw_2013` |

---

## 5. Results (Scope 1, Pacific, 2014-2025)

### 5.1 Headline performance summary

| Portfolio | Ann. return | Ann. vol | Ann. RF | Sharpe | TE vs VW | Min monthly | Max monthly | Cumulative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vw`       |  6.72 % | 12.88 % | 1.75 % | 0.39 |   —    | -10.76 % | 13.07 % | 118.34 % |
| `vw_drift` |  6.62 % | 13.19 % | 1.75 % | 0.37 | 1.45 % | -11.95 % | 12.98 % | 115.72 % |
| `mv`       |  8.30 % | 11.58 % | 1.75 % | 0.57 | 9.07 % |  -8.08 % | 11.04 % | 160.32 % |
| `mv_50`    |  9.06 % | 11.70 % | 1.75 % | 0.62 | 9.01 % |  -8.14 % | 10.63 % | 183.14 % |
| `vw_50`    |  7.79 % | 13.59 % | 1.75 % | 0.44 | 3.24 % | -14.40 % | 11.59 % | 146.02 % |
| `vw_nz`    |  7.70 % | 13.63 % | 1.75 % | 0.44 | 3.16 % | -14.29 % | 11.95 % | 143.65 % |

(Source: `outputs/part3_performance_summary.csv` and `outputs/part4_performance_summary.csv`. Sharpe = (ann return − ann RF) / ann vol with the
annualised Pacific RF computed as `rf.mean() * 12`, matching `src/MVP-construction.ipynb`. Tracking
error is the annualised std of `R_p − R_vw` over the full 2014-01 → 2025-12 window.)

### 5.2 Carbon-constraint check

Maximum excess of `CF` above its `carbon_limit` over the 12 years:

| Portfolio | Max excess (tCO₂e / MUSD) | Mean excess |
|---|---:|---:|
| `mv_50` | 0.000 | 0.000 |
| `vw_50` | 0.000 | -1.475 |
| `vw_nz` | 0.000 | -3.377 |

All three constrained portfolios respect their caps each year. `vw_50`
and `vw_nz` are at or slightly below the cap (the slack comes from
years where the binding constraint is feasibility within the eligible
universe, not the carbon cap).

### 5.3 Investment-universe size

| Year | n eligible | | Year | n eligible |
|---|---:|---|---|---:|
| 2013 | 259 | | 2019 | 403 |
| 2014 | 269 | | 2020 | 432 |
| 2015 | 290 | | 2021 | 462 |
| 2016 | 314 | | 2022 | 474 |
| 2017 | 341 | | 2023 | 472 |
| 2018 | 365 | | 2024 | 469 |

The universe roughly doubles between 2013 and 2024 as carbon coverage
expands.

### 5.4 Output files for the report

In `outputs/` (Part III — from `src/05_part3.py`):
* `part3_performance_summary.csv` — RF-adjusted Sharpe + TE (`vw`, `vw_drift`, `mv`, `mv_50`, `vw_50`)
* `part3_annual_carbon_metrics.csv`
* `part3_monthly_returns.csv`
* `part3_portfolio_weights.csv`
* `part3_top10_waci_drivers_by_year.csv`
* `part3_top10_cf_drivers_by_year.csv`
* `part3_tracking_error.csv` — ex-ante and ex-post annual TE per portfolio
* `part3_optimizer_status.csv` — SLSQP success/message per year × portfolio
* `part3_carbon_constraint_slack.csv` — limit, realised CF, slack per year × portfolio
* `part3_exclusions_overweights.csv`

In `outputs/` (Part IV — from `src/06_part4.py`):
* `part4_performance_summary.csv` — RF-adjusted Sharpe + TE (`vw`, `vw_drift`, `vw_nz`)
* `part4_annual_carbon_metrics.csv`
* `part4_monthly_returns.csv`
* `part4_portfolio_weights.csv`
* `part4_netzero_path.csv` — anchor CF, cap target, realised CF(vw_nz), CF(vw) per year
* `part4_tracking_error.csv`
* `part4_optimizer_status.csv`
* `part4_carbon_constraint_slack.csv`
* `part4_exclusions_overweights.csv`

In `outputs/figures/`:
* `part3_1_waci.png` — annual WACI trajectory
* `part3_1_carbon_footprint.png` — annual CF trajectory
* `part3_2_mv_vs_mv50_cumulative.png` — VW vs MV vs MV(0.5) (starts at $1)
* `part3_3_vw_vs_vw50_cumulative.png` — VW vs VW(0.5) (starts at $1)
* `part3_4_tracking_error.png` — annual ex-ante TE per Part III constrained portfolio
* `part4_1_netzero_cumulative.png` — VW vs VW(NZ) (starts at $1)
* `part4_2_netzero_path.png` — net-zero target cap vs realised CF

---

## 6. Interpretation (for the report)

* `mv` already beats `vw` on both return and Sharpe. **However**, its
  carbon footprint is structurally **higher** than the benchmark in
  every year, because min-variance picks stable, capital-intensive
  firms (utilities, materials, mining) that also happen to be high CI.
  → A 50 % cut on `mv` is a real, economically meaningful constraint.
* `mv_50` ends up with a **higher Sharpe than `mv`** in this sample.
  Forcing the optimiser away from high-carbon firms also pushes it
  toward more diversified industries; in the Pacific universe this
  was beneficial out of sample.
* `vw_50` cuts the benchmark CF in half at modest cost: +0.7 pp
  volatility, +1 pp annualised return.
* `vw_nz` follows a tightening path. Cap roughly equals `vw_50`'s
  early on; by 2024 it is `0.9^12 ≈ 28 %` of `CF(vw)_{2013}`. The
  realised return tracks `vw_50` closely; Sharpe is essentially
  identical.

### Things to double-check before submitting

1. **Group strategy.** The script assumes **Scope 1**. If your
   group's assigned strategy is Scope 2 or Scope 1+2, change
   `CARBON_SCOPE` at the top of `src/saam_core.py` and rerun.
   Every output regenerates automatically.
2. **Region.** Only Pacific is loaded. If your region is something
   else, that has to be changed earlier (in `src/01_cleaning.py`).
3. **Risk-free rate.** Sharpe ratios now use the Pacific monthly RF
   loaded from `data/processed/rf_rate.csv` (RF values in percent per
   month; the engine divides by 100 and reports `ann_rf = rf.mean() * 12`).
   To reproduce a 0 % Sharpe, pass `rf=None` to `summary_stats` in
   `src/05_part3.py:run` and `src/06_part4.py:run`.
4. **Report length.** PDF Section 5.2 caps the report at 30 pages.
   The deliverable bundle is: report (PDF), 1-page sales pitch (PDF),
   the notebook, the video.
5. **Mandatory LLM disclosure.** The required paragraph is in the
   notebook (Section 10) and in `documentation/part3_4_methodology.md`.

---

## 7. Are any results "wrong" and need changing?

After re-checking against the PDF formulas (Sections 2–4):

| Item | Status |
|---|---|
| Carbon-footprint formula `Σ w_i · E_i / Cap_i` | ✅ matches PDF |
| Carbon-intensity formula `E_i / (Rev_i/1000)` | ✅ matches PDF (Rev given in kUSD) |
| WACI formula `Σ w_i · CI_i` | ✅ matches PDF |
| Net-zero exponent `(1-θ)^(Y-Y0+1)` | ✅ matches PDF (anchored on `cf_vw_2013`) |
| VW benchmark: monthly rebalancing | ✅ added in this round, strict PDF 2.3 |
| Eligibility filters | ✅ history ≥ 36 months, stale ≤ 50 %, carbon present at end-Y, price > 0 |
| Optimiser convergence | ✅ all carbon caps satisfied within 1e-6 each year |
| Drift formula for performance | ✅ matches PDF 2.2 |

Nothing requires changing on numerical grounds. The remaining
**discretionary** choices (any of which a teammate may want to vary) are:

* **Scope.** Switch `CARBON_SCOPE` to test Scope 2 or Scope 1+2.
* **Estimation window.** 120 months is the PDF default; a smaller
  window (e.g. 60) is a common robustness check.
* **Eligibility thresholds.** `MIN_OBS` and `STALE_THRESHOLD` are at
  the values suggested by the PDF; tightening either reduces the
  universe and is also a useful robustness check.

---

## 8. Where to look if something breaks

| Symptom | First place to look |
|---|---|
| File not found error | `data/processed/` is empty → rerun `src/01_cleaning.py` and `src/02_analysis.py` first |
| Carbon cap violated | print `cf_per_w.describe()` for that year — a single firm with extreme `e_i` can dominate |
| `mv` return looks wrong | check Σ conditioning: `np.linalg.eigvalsh(cov).min()` should be > `RIDGE_EPS` |
| Empty universe in year Y | one of the filters is too tight; print `eligible.value_counts()` after each step |
| Numpy/Pandas import error | `pip install -r requirements.txt` from the project root |

---

## Use of Large Language Models (LLMs)

We used Claude (Anthropic) and ChatGPT (OpenAI) as coding-support
tools for clarifying algebra, debugging pandas/numpy, suggesting
SLSQP over a hand-rolled projection, and polishing this note. All
methodological choices, the optimisation problem statements, the
simulation logic and every interpretation are the group's own work.
The group is fully responsible for correctness and academic integrity.

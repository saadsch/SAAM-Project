# Parts III & IV — methodology and review of the ChatGPT implementation

This note documents the carbon-aware allocation work for the SAAM 2026 project
and explains exactly what was changed relative to the previous ChatGPT-generated
implementation.

## 1. What the project does (refresher)

The SAAM 2026 handout (`documentation/SAAM_Project_2026 (1).pdf`) asks each
group to construct several long-only equity portfolios on the assigned region
(Pacific) over 2014–2025, with annual rebalancing decided at the end of each
year:

| Portfolio  | Optimisation problem | Constraint |
|------------|---------------------|------------|
| `vw`       | Value-weighted benchmark | weights = market caps |
| `mv`       | min wᵀΣw                  | w ≥ 0, Σw = 1 |
| `mv_50`    | min wᵀΣw                  | w ≥ 0, Σw = 1, CF(w) ≤ 0.5 · CF(`mv`) |
| `vw_50`    | min (w−w_vw)ᵀΣ(w−w_vw)    | w ≥ 0, Σw = 1, CF(w) ≤ 0.5 · CF(`vw`) |
| `vw_nz`    | min (w−w_vw)ᵀΣ(w−w_vw)    | w ≥ 0, Σw = 1, CF(w) ≤ (1−θ)^(Y−2013+1) · CF(`vw`)_{2013}, θ=10% |

Carbon definitions used:

* **Carbon footprint of a portfolio** `CF(w) = Σ_i (V_i / Cap_i) · E_i / V = Σ_i w_i · E_i / Cap_i`, in tCO₂e per USD-million invested.
* **WACI of a portfolio** `WACI(w) = Σ_i w_i · CI_i`, where `CI_i = E_i / (Rev_i/1000)` in tCO₂e per USD-million revenue.

Performance is computed month by month from January 2014 to December 2025;
weights drift between rebalancing dates as in PDF Section 2.2.

## 2. Files in the project

* `src/01_cleaning.py` — extracts the Pacific universe from the Datastream
  files and writes the cleaned annual panels (`Clean_*.csv`) and the price
  history.
* `src/02_analysis.py` — implements the price-cleaning rules from the handout
  (price floor at 0.5, forward-fill middle gaps, delisting = −100% return),
  applies the 36-month and 50%-stale filters, checks carbon coverage and
  produces `data/processed/final_returns_matrix.csv` plus the universe
  snapshot.
* `src/03_outputs.py` — Part I figures (risk vs CI scatter, descriptive
  statistics).
* `src/saam_core.py` — shared helpers (data loading, cleaning, SLSQP, simulation, stats).
* `src/05_part3.py` — Part III optimisation engine with explicit `# === Section X.Y === #` markers (3.1, 3.2, 3.3, 3.4).
* `src/06_part4.py` — Part IV (Net-Zero) optimisation engine with `# === Section 4.X === #` markers (4.1, 4.2).
* `src/Part3_Part4_notebook.ipynb` — single reproducible notebook for the
  final-deliverable requirement of the PDF.
* `notebooks/`, `MVP-construction.ipynb`, `vwp.ipynb` — exploratory Part I
  notebooks.

## 3. Methodology used in `src/05_part3.py`, `src/06_part4.py` and `src/saam_core.py`

* **Solver**: `scipy.optimize.minimize` with the SLSQP method. The
  optimisation problems are now passed in their natural form
  (quadratic objective, linear equality + inequality constraints, box
  bounds), with closed-form gradients. SLSQP is a recognised SQP method
  and is fully appropriate for convex QPs of this size (~500 variables).
* **Initialisation**: 1/n for the variance objective; **benchmark `w_vw`
  for the tracking-error objective**, optionally mixed with the
  lowest-carbon vertex of the simplex if the cap is violated. This is the
  fix that makes `vw_50` and `vw_nz` sit *on* the carbon boundary (where
  the constrained optimum lies, since the unconstrained TE optimum is
  `w = w_vw` and violates the cap whenever the cap is below the benchmark
  footprint).
* **Covariance estimation**: complete-case rows of the 120-month window,
  with a fallback to the mean-fill rule when fewer than 36 complete rows
  are available. A small ridge `1e-6 · I` is added for numerical safety.
* **Value-weighted benchmark performance**: the strict PDF reading
  (Section 2.3) is monthly rebalancing using the end-of-month
  capitalisation file (`DS_MV_T_USD_M.csv`). The new script reports
  both `vw` (monthly rebalanced) and `vw_drift` (end-of-year caps, drifts
  within the year, used as benchmark vector for tracking error). The
  difference is small but non-zero.
* **Composition diagnostics**: a new output
  `outputs/part3_exclusions_overweights.csv` lists, for every year and
  every constrained portfolio, the ten largest exclusions vs the
  benchmark and the ten largest overweights vs the benchmark. This
  answers the PDF's request for "main changes regarding the composition
  of the portfolio" in Section 3.4.
* **Risk-free rate**: Sharpe is computed with the Pacific monthly
  risk-free series in `data/processed/rf_rate.csv` (RF in percent per
  month; the engine divides by 100 and reports
  `ann_rf = rf.mean() * 12`, matching `src/MVP-construction.ipynb`).
* **Tracking error**: per allocation year and overall, the engine writes
  the annualised ex-ante TE
  `sqrt((w − w_vw)' Σ_Y (w − w_vw) · 12)`
  and the annualised ex-post TE
  `std(R_p − R_vw_drift) · √12` over the 12 implementation months.
  The full-sample TE against the monthly-rebalanced VW also appears in
  `outputs/part3_performance_summary.csv` (`tracking_error_vs_vw`).
* **Other new diagnostics**:
  `outputs/part3_top10_cf_drivers_by_year.csv` (top contributors to
  CF(vw)_Y), `outputs/part3_optimizer_status.csv` (one row per year ×
  portfolio with SLSQP success and message),
  `outputs/part3_carbon_constraint_slack.csv` (limit − realised per
  year × portfolio).
* **Net-zero diagnostic plot**: `outputs/figures/part4_netzero_path.png`
  overlays the net-zero target trajectory and the realised footprint of
  `vw_nz` and `vw_drift`, which is a more readable presentation than the
  generic "CF by portfolio" chart for the Section 4.2 commentary.

## 5. Headline results (Scope 1, Pacific, 2014–2025)

| Portfolio  | Ann. return | Ann. vol | Ann. RF | Sharpe | TE vs VW |
|------------|------------:|---------:|--------:|-------:|---------:|
| `vw`       |  6.72%      | 12.88%   | 1.75%   | 0.39   |   —      |
| `vw_drift` |  6.62%      | 13.19%   | 1.75%   | 0.37   | 1.45%    |
| `mv`       |  8.30%      | 11.58%   | 1.75%   | 0.57   | 9.07%    |
| `mv_50`    |  9.06%      | 11.70%   | 1.75%   | 0.62   | 9.01%    |
| `vw_50`    |  7.79%      | 13.59%   | 1.75%   | 0.44   | 3.24%    |
| `vw_nz`    |  7.70%      | 13.63%   | 1.75%   | 0.44   | 3.16%    |

Sharpe uses the annualised Pacific RF (`rf.mean() * 12` over the
performance window, RF values from `data/processed/rf_rate.csv` in
percent per month). Tracking error is the annualised std of `R_p − R_vw`.

All five constrained portfolios respect their carbon caps each year
(verified in the notebook: the maximum excess of `CF` above `carbon_limit`
is below 1 t/M\$ for every portfolio).

## 6. Interpretation prompts for the report

* `mv` is more carbon-heavy than `vw` in every year of the Pacific
  sample: minimum-variance picks stable, capital-intensive firms
  (utilities, materials, mining), which are also high-CI. The carbon
  reduction in `mv_50` is therefore real, not cosmetic.
* `vw_50` reaches the 50% cut while keeping a low tracking error to the
  benchmark — the cost is roughly +0.7 pp of annualised volatility for
  +1 pp of annualised return relative to `vw`.
* `vw_nz` follows a tightening path. The cap is loose enough in the
  first half of the sample that `vw_nz` and `vw_50` deliver similar
  carbon. In the second half the cap binds harder and the realised
  return drifts slightly above `vw_50` thanks to favourable tilts toward
  lower-carbon segments.
* `outputs/part3_exclusions_overweights.csv` and
  `outputs/part3_top10_waci_drivers_by_year.csv` make it easy to
  point to the specific firms that drive these tilts each year.

## 7. Limitations to discuss in the report

* Imperfect, forward-filled carbon coverage.
* Estimation error in $\Sigma_Y$ — long-only QP weights are sensitive
  to it, especially on short-history firms.
* The Scope 1 assumption excludes indirect emissions; Scope 2 and
  Scope 1+2 alternatives are available via the `CARBON_SCOPE` switch.
* Carbon constraints can create sector/country tilts. Sector
  concentration is not penalised in the optimisation; this would be
  the next robustness check.
* The net-zero trajectory anchors on the 2013 VW footprint; results
  depend on that base year and on the value of $\theta$.

## Use of Large Language Models (LLMs)

Claude (Anthropic) and ChatGPT (OpenAI) were used as coding-support
tools to: clarify the algebra of the constraints in the handout, debug
pandas/numpy indexing, identify that the original projected-gradient
solver was stopping after two iterations, and polish the language of
this note. All methodological choices, the implementation of the
optimisation problems, the simulation logic and the interpretation of
the results are the group's own work. The group is fully responsible
for correctness and academic integrity.

# Recommendations for Part 1 & Part 2

These notes are for the collaborators who own Part 1 (cleaning + MVP) and
Part 2 (value-weighted benchmark). **Do not regenerate Part 1/2 outputs to
implement these - they are advisory.** They matter because Parts 3 and 4
consume the cleaned panels (`data/processed/Clean_*_Pacific.csv`,
`Pacific_Universe.csv`, `DS_MV_T_USD_*.csv`) and the eligibility logic.

## What to check in Part 1 (`src/01_cleaning.py`, `MVP-construction.ipynb`)

1. **Price cleaning rules (PDF Section 2.1).** Confirm in
   `01_cleaning.py` that:
   - prices < 0.5 are set to NaN,
   - middle missing values are forward-filled,
   - trailing missing values are set to 0 (so the delisting month returns
     -100%, not NaN).
   Part 3/4 re-applies this rule defensively in `saam_core._clean_prices`,
   so a mismatch would silently change the universe between parts.

2. **Eligibility filter.** Part 1 should require at every end-Y:
   `price > 0`, `>= 36 monthly observations in the 120-month estimation
   window`, `<= 50% zero returns in that window`, and non-missing
   emissions / cap / revenues at Y. Confirm the thresholds match those used
   in Parts 3 / 4 (`MIN_OBS=36`, `STALE_THRESHOLD=0.5`, `ESTIM_MONTHS=120`
   in `saam_core.py`). Any change will shift the eligible universe.

3. **MV optimiser (PDF Section 2.2).** Verify the solver is long-only
   (`w >= 0`), enforces `sum(w) = 1`, and uses the 120-month window. Part 3
   uses SLSQP via `saam_core.solve_qp(objective="variance")` - results
   should match the Part 1 MVP at the unconstrained step.

4. **Risk-free rate Sharpe.** Confirm Sharpe in `MVP-construction.ipynb`
   uses the Pacific RF from `data/processed/rf_rate.csv` (RF in
   percent-per-month, divided by 100, annualised as `mean * 12`). Parts 3
   and 4 use this exact convention - mismatched conventions across the
   project would make headline tables inconsistent.

## What to check in Part 2 (`src/vwp.ipynb`, `02_analysis.py`,
`03_outputs.py`)

1. **Monthly rebalancing (PDF Section 2.3).**
   `R^{vw}_{t+1} = sum_i (Cap_{i,t} / sum_j Cap_{j,t}) * R_{i,t+1}`.
   In pandas this is `(cap.shift(1) / cap.shift(1).sum(axis=1)) * rets`.
   The `shift(1)` is critical (no look-ahead). Part 3/4 implements this in
   `saam_core.vw_monthly_rebalanced` - if Part 2 uses a different
   convention (e.g. same-month cap, annual cap), the VW benchmark series
   between parts will not align.

2. **Universe coherence.** The VW benchmark should be computed on the same
   eligible universe (or at least the same Pacific universe) as the MVP.
   Part 3/4 takes the intersection of `Pacific_Universe.csv`,
   `Clean_*_Pacific.csv`, `DS_MV_T_USD_Y.csv`, `DS_MV_T_USD_M.csv`. If
   Part 2 includes ISINs that are excluded by Part 1/3, the VW curve will
   be biased.

3. **Currency / units.** Cap files (`DS_MV_T_USD_*`) must be in USD
   millions, revenue files in thousands USD. The Part 3/4 carbon-footprint
   formula assumes those exact units (CI = E / (Rev/1000), CF_per_w =
   E / Cap_USD_millions). A unit mismatch silently shifts the magnitudes
   of all WACI/CF outputs.

## Suspicious / latent issues found

- **No look-ahead enforced in Part 2 if it currently uses end-of-year cap
  weights and same-year returns.** Verify the `shift(1)` is in place.
- **Mean-fill vs complete-case covariance.** Part 3/4 falls back to
  mean-filled covariance when fewer than 36 firms have full 120-month
  histories. Confirm Part 1 MVP uses the same fallback - otherwise the
  unconstrained MV portfolio in Part 1 and Part 3 will disagree.
- **Risk-free CSV column convention.** `data/processed/rf_rate.csv` has
  header `YYYYMM, RF` with RF in %/month. Any notebook that treats RF as
  an annual figure or as fraction-per-month will be wrong by a factor of
  100 or 12.
- **Annual rebalancing semantics.** PDF Section 2.2 says weights are
  computed at end-Y and held through Y+1 with drift. Confirm Part 1 MVP
  outputs follow that, not monthly rebalancing of MV.

## Why this matters for Parts 3 and 4

- The 50%-cap target in Part 3.2 is `0.5 * CF(P_mv_oos)`. If Part 1 MVP
  weights differ from Part 3's unconstrained MV solve, the cap used by
  Part 3 (which recomputes MV internally) will differ from the cap
  expected by Part 1 - reconciliation across the report becomes confusing.
- The Net-Zero anchor in Part 4 is `CF(P_vw)_2013`. If Part 2 VW differs
  from Part 3's VW monthly-rebalanced benchmark, the anchor and target
  path will not match the rest of the report.
- Cleaning rules feed every downstream return, cap and CF calculation.
  Any silent change to `Clean_Prices_Pacific.csv` will shift Part 3 and
  Part 4 outputs.

## Recommendation

Do not change Part 1/2 code or outputs at this stage. After the final
report, align on **one** canonical implementation of:

1. Price-cleaning rules.
2. Eligibility filter (`MIN_OBS`, `STALE_THRESHOLD`, `ESTIM_MONTHS`).
3. MV solver settings.
4. VW formula with `shift(1)`.
5. RF convention.

Then re-run all four parts top-to-bottom. The Part 3/4 scripts already
read the cleaned panels directly and will pick up the canonical version
automatically.

"""
SAAM Project — Parts III and IV (carbon-aware allocations).

Solver
------
The constrained quadratic programs are solved with `scipy.optimize.minimize`
using SLSQP (a standard sequential-quadratic-programming solver). The
problem statements are the ones described in the SAAM 2026 PDF
(Sections 3.2, 3.3 and 4):

  - the constraints (sum to one, non-negative, carbon cap) are passed
    explicitly to a recognised QP solver,
  - analytic gradients are provided so SLSQP converges in a controllable
    way,
  - the covariance matrix is estimated on complete-case rows of the
    rolling 120-month window with a small ridge for numerical stability,
  - the value-weighted benchmark is computed two ways:
      * monthly rebalancing using monthly caps (for the *performance*
        comparison, Section 2.3),
      * annual end-of-year capitalisations (for the *tracking-error*
        constraint, Sections 3.3 and 4.1).

Inputs (already produced by `src/01_cleaning.py`):
  - data/processed/Clean_Prices_Pacific.csv
  - data/processed/Clean_Revenues_Pacific.csv
  - data/processed/Clean_CO2_Scope1_Pacific.csv
  - data/processed/Pacific_Universe.csv
  - data/raw/DS_MV_T_USD_Y.csv
  - data/raw/DS_MV_T_USD_M.csv

Outputs (all under `outputs/`):
  - part3_4_monthly_portfolio_returns.csv
  - part3_4_performance_summary.csv
  - part3_4_annual_carbon_metrics.csv
  - part3_4_portfolio_weights.csv
  - part3_4_top10_waci_drivers_by_year.csv
  - part3_4_exclusions_overweights.csv
  - figures/part3_mv_vs_mv50_cumulative.png
  - figures/part3_vw_vs_vw50_cumulative.png
  - figures/part4_vw_vw50_netzero_cumulative.png
  - figures/part3_4_waci_by_portfolio.png
  - figures/part3_4_carbon_footprint_by_portfolio.png
  - figures/part4_netzero_path.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
RAW = DATA / "raw"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

# Annual allocation years: end of YEAR_FIRST .. end of YEAR_LAST.
# Implementation runs the year after the allocation decision.
YEAR_FIRST = 2013
YEAR_LAST = 2024
PERF_START = 2014
PERF_END = 2025

ESTIM_MONTHS = 120          # 10 years of monthly returns
MIN_OBS = 36                # >= 3 years available within the window
STALE_THRESHOLD = 0.50      # at most 50% zero returns in the window
STARTING_WEALTH_MUSD = 1.0  # V_2013 = USD 1m

CARBON_SCOPE = "scope1"     # group's assigned scope; switch to "scope2" or
                            # "scope1_plus_scope2" if the strategy changes
THETA_NZ = 0.10             # 10% annual carbon-footprint reduction

RIDGE_EPS = 1e-6            # small ridge added to Sigma for numerical safety


# --------------------------------------------------------------------------- #
# Data loading helpers                                                        #
# --------------------------------------------------------------------------- #

def _read_wide(path: Path, date_columns: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Read a Datastream-style wide CSV.

    Layout: one row per firm, identified by ISIN; the first columns are
    NAME / ISIN, then one column per period (year or month-end date).
    We return:
      df    -- DataFrame indexed by ISIN with date columns (numeric).
      names -- Series ISIN -> NAME (the firm display name).
    """
    df = pd.read_csv(path)
    df = df[df["ISIN"].notna()].copy()
    df["ISIN"] = df["ISIN"].astype(str)
    # Keep the names side-table separately so they survive the numeric cast.
    names = df.set_index("ISIN")["NAME"].astype(str) if "NAME" in df.columns else pd.Series(dtype=str)
    df = df.set_index("ISIN").drop(columns=["NAME"], errors="ignore")
    # Coerce every cell to a float, turning Datastream's blanks/strings into NaN.
    df = df.apply(pd.to_numeric, errors="coerce")
    # Normalise column labels: full timestamps for monthly panels, plain ints for annual panels.
    df.columns = pd.to_datetime(df.columns).normalize() if date_columns else df.columns.astype(int)
    return df, names


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Apply the PDF's price cleaning rules:
    - prices below 0.5 are treated as missing,
    - middle missing values are forward-filled,
    - trailing missing values (delisting) are set to 0, so the realised
      monthly return on delisting equals -100%.
    Result: panel of dates x ISINs (columns).
    """
    # Transpose so rows become dates and columns become ISINs (easier to compute returns).
    prices = prices.T.sort_index()
    # PDF rule: ignore implausibly small prices (rounding artefacts in Datastream).
    prices[prices < 0.5] = np.nan
    # Build a mask that says, for every cell, whether ANY later date has a valid price.
    # If ahead == 0 the cell is in a trailing gap: the firm was delisted.
    has = prices.notna().astype(np.int8)
    ahead = has.iloc[::-1].cumsum(axis=0).iloc[::-1]
    trailing = prices.isna() & (ahead == 0)
    # Middle gaps: forward-fill (PDF rule, "use the number from the previous year").
    prices = prices.ffill(axis=0)
    # Delisting: set the price to exactly 0 so the next computed return is -100%.
    prices[trailing] = 0.0
    return prices


def _returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly returns with delisting handled as -100%."""
    prev = prices.shift(1)
    # Standard simple return P_t/P_{t-1} - 1.  Avoid division by 0 (delisting month).
    rets = prices / prev.replace(0.0, np.nan) - 1.0
    # If the firm just delisted (P_t = 0 while P_{t-1} > 0) the return is exactly -100%.
    rets[(prices == 0.0) & (prev > 0.0)] = -1.0
    return rets.replace([np.inf, -np.inf], np.nan)


def _annual_panel(path: Path) -> pd.DataFrame:
    """Annual wide panel forward-filled across years (per the PDF rule on
    missing carbon/revenue entries)."""
    df, _ = _read_wide(path, date_columns=False)
    return df.sort_index(axis=1).ffill(axis=1)


def _load_inputs():
    prices_raw, names_p = _read_wide(PROCESSED / "Clean_Prices_Pacific.csv", date_columns=True)
    prices = _clean_prices(prices_raw)
    rets = _returns(prices)

    scope1 = _annual_panel(PROCESSED / "Clean_CO2_Scope1_Pacific.csv")
    if CARBON_SCOPE == "scope1":
        emissions = scope1
    elif CARBON_SCOPE == "scope2":
        emissions = _annual_panel(PROCESSED / "Clean_CO2_Scope2_Pacific.csv")
    elif CARBON_SCOPE == "scope1_plus_scope2":
        scope2 = _annual_panel(PROCESSED / "Clean_CO2_Scope2_Pacific.csv")
        emissions = scope1.add(scope2, fill_value=0.0)
    else:
        raise ValueError(f"Unsupported CARBON_SCOPE={CARBON_SCOPE!r}")

    revenues = _annual_panel(PROCESSED / "Clean_Revenues_Pacific.csv")

    static = pd.read_csv(PROCESSED / "Pacific_Universe.csv")
    name_col = "NAME" if "NAME" in static.columns else "Name"
    names = static.set_index("ISIN")[name_col].astype(str)
    countries = static.set_index("ISIN")["Country"].astype(str)

    pacific = set(static["ISIN"].astype(str))
    cap_y = _annual_panel(RAW / "DS_MV_T_USD_Y.csv")
    cap_y = cap_y.loc[cap_y.index.intersection(pacific)]

    cap_m_raw, _ = _read_wide(RAW / "DS_MV_T_USD_M.csv", date_columns=True)
    cap_m = cap_m_raw.T.sort_index()
    cap_m = cap_m.loc[:, cap_m.columns.intersection(list(pacific))]

    common = (
        prices.columns
        .intersection(emissions.index)
        .intersection(revenues.index)
        .intersection(cap_y.index)
    )
    prices = prices[common]
    rets = rets[common]
    emissions = emissions.loc[common]
    revenues = revenues.loc[common]
    cap_y = cap_y.loc[common]
    cap_m = cap_m.loc[:, cap_m.columns.intersection(common)]

    return prices, rets, emissions, revenues, cap_y, cap_m, names, countries


# --------------------------------------------------------------------------- #
# Investment set per year                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class YearInputs:
    year: int
    isins: list[str]
    names: pd.Series
    countries: pd.Series
    rets_est: pd.DataFrame
    rets_oos: pd.DataFrame
    mu: np.ndarray
    cov: np.ndarray
    cap_y: pd.Series
    emissions: pd.Series
    revenues: pd.Series
    ci: pd.Series                 # carbon intensity in tCO2e per USD-million revenue
    cf_per_w: pd.Series           # E_i / Cap_i  -> contribution to footprint per unit weight
    w_vw: np.ndarray              # annual end-of-year value-weighted weights


def _build_year_inputs(year, prices, rets, emissions, revenues, cap_y, names, countries) -> YearInputs:
    # The PDF defines the allocation decision at the end of year `year`,
    # using a trailing 10-year window. Implementation runs over `year + 1`.
    est_start = pd.Timestamp(year - 9, 1, 1)
    est_end = pd.Timestamp(year, 12, 31)
    oos_start = pd.Timestamp(year + 1, 1, 1)
    oos_end = pd.Timestamp(year + 1, 12, 31)

    rets_window_all = rets.loc[est_start:est_end]
    rets_oos_all = rets.loc[oos_start:oos_end]

    # Per-firm history quality metrics over the estimation window.
    obs = rets_window_all.count()                                 # # of non-NaN monthly returns
    zero_ratio = (rets_window_all == 0.0).sum() / obs.replace(0, np.nan)  # share of zero returns
    last_price_valid = prices.loc[:est_end].iloc[-1].gt(0)        # price strictly positive at end-Y

    # Eligibility = every filter from PDF Sections 1 + 2.1:
    #   1. carbon, revenue, market cap available at end-Y;
    #   2. last observed price > 0 (no trailing delisting at decision date);
    #   3. enough history to estimate Sigma (≥ 36 monthly observations);
    #   4. not stale (≤ 50% zero returns over the window).
    eligible = (
        emissions[year].notna()
        & revenues[year].gt(0)
        & cap_y[year].gt(0)
        & last_price_valid.reindex(emissions.index).fillna(False)
        & obs.reindex(emissions.index).ge(MIN_OBS).fillna(False)
        & zero_ratio.reindex(emissions.index).le(STALE_THRESHOLD).fillna(False)
    )
    isins = sorted(eligible[eligible].index.intersection(rets_oos_all.columns))
    if not isins:
        raise RuntimeError(f"No eligible firms for year {year}")

    # Restrict to the eligible universe and clip to ESTIM_MONTHS (=120).
    rets_est = rets_window_all[isins].iloc[-ESTIM_MONTHS:]
    rets_oos = rets_oos_all[isins]

    # Covariance estimation.
    # Preferred: complete-case rows (rows with no missing value across the
    # eligible firms). This avoids the bias of replacing NaN with column
    # means before computing covariances. We fall back to the mean-fill
    # rule only if too few complete rows are available.
    complete = rets_est.dropna(how="any")
    if len(complete) < MIN_OBS:
        filled = rets_est.apply(lambda s: s.fillna(s.mean()), axis=0).fillna(0.0)
        mu = filled.mean().to_numpy()
        cov = np.cov(filled.to_numpy(), rowvar=False, ddof=0)
    else:
        mu = complete.mean().to_numpy()
        cov = np.cov(complete.to_numpy(), rowvar=False, ddof=0)
    # Symmetrise (numerical noise) and add a small ridge so Sigma is strictly PD.
    cov = (cov + cov.T) / 2.0
    cov += RIDGE_EPS * np.eye(cov.shape[0])

    # Aligned annual quantities used by the optimiser and the carbon metrics.
    cap_yi = cap_y.loc[isins, year].astype(float)
    emi_yi = emissions.loc[isins, year].astype(float)
    rev_yi = revenues.loc[isins, year].astype(float)

    # PDF formulas:
    #   Carbon intensity:  CI_i = E_i / (Rev_i / 1000)   (Rev is in '000 USD; we want USD million.)
    #   CF per unit weight: e_i = E_i / Cap_i           (since CF(w) = Σ w_i · E_i / Cap_i.)
    ci = emi_yi / (rev_yi / 1000.0)
    cf_per_w = emi_yi / cap_yi
    # Value-weighted reference for tracking error and for the benchmark itself.
    w_vw = (cap_yi / cap_yi.sum()).to_numpy()

    return YearInputs(
        year=year,
        isins=isins,
        names=names.reindex(isins),
        countries=countries.reindex(isins),
        rets_est=rets_est,
        rets_oos=rets_oos,
        mu=mu,
        cov=cov,
        cap_y=cap_yi,
        emissions=emi_yi,
        revenues=rev_yi,
        ci=ci,
        cf_per_w=cf_per_w,
        w_vw=w_vw,
    )


# --------------------------------------------------------------------------- #
# Optimisation: SLSQP-based long-only QP                                       #
# --------------------------------------------------------------------------- #

def _solve_qp(
    cov: np.ndarray,
    objective: str = "variance",
    benchmark: np.ndarray | None = None,
    carbon: np.ndarray | None = None,
    carbon_limit: float | None = None,
) -> tuple[np.ndarray, bool, str]:
    """Solve the long-only constrained QP via SLSQP.

    objective="variance"        :   min  w' Sigma w
    objective="tracking_error"  :   min (w - benchmark)' Sigma (w - benchmark)
    subject to:
        sum(w) = 1
        0 <= w_i <= 1
        carbon . w <= carbon_limit   (if carbon is provided)
    """
    n = cov.shape[0]
    if benchmark is None:
        benchmark = np.full(n, 1.0 / n)

    # Initial point.
    # - For the variance objective the unconstrained optimum is unknown, so
    #   start at 1/n (interior of the simplex).
    # - For the tracking-error objective the unconstrained optimum is the
    #   benchmark itself, so start from the benchmark whenever it is
    #   feasible; otherwise mix it with the lowest-carbon vertex of the
    #   simplex to obtain a feasible warm-start that is still as close to
    #   the benchmark as possible.
    if objective == "tracking_error":
        x0 = np.asarray(benchmark, dtype=float).copy()
    else:
        x0 = np.full(n, 1.0 / n)

    if carbon is not None and carbon_limit is not None:
        if float(x0 @ carbon) > carbon_limit:
            lo_i = int(np.argmin(carbon))
            unit = np.zeros(n)
            unit[lo_i] = 1.0
            base = float(x0 @ carbon)
            low = float(unit @ carbon)
            if low > carbon_limit:
                # No firm alone meets the cap: problem is infeasible.
                x0 = unit
            else:
                # Smallest mix with the low-carbon vertex that satisfies the
                # cap exactly, plus a tiny slack so SLSQP starts strictly
                # feasible.
                alpha = (base - carbon_limit) / max(base - low, 1e-12)
                alpha = float(np.clip(alpha + 1e-3, 0.0, 1.0))
                x0 = (1.0 - alpha) * x0 + alpha * unit

    # Objective and analytic gradient. Passing the gradient explicitly is
    # important: SLSQP's finite-difference fallback would scale poorly with n.
    if objective == "variance":
        def fun(w):
            return float(w @ cov @ w)
        def jac(w):
            return 2.0 * cov @ w
    elif objective == "tracking_error":
        b = np.asarray(benchmark, dtype=float)
        def fun(w):
            d = w - b
            return float(d @ cov @ d)
        def jac(w):
            return 2.0 * cov @ (w - b)
    else:
        raise ValueError(objective)

    # Equality constraint: weights sum to 1 (full-investment, no cash).
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
                    "jac": lambda w: np.ones_like(w)}]
    # Optional carbon-cap inequality: c·w ≤ L. SLSQP requires "ineq" functions
    # to return a value that should stay >= 0, so we pass L - c·w.
    if carbon is not None and carbon_limit is not None:
        c = np.asarray(carbon, dtype=float)
        constraints.append({
            "type": "ineq",
            "fun": lambda w, c=c, L=float(carbon_limit): float(L - w @ c),
            "jac": lambda w, c=c: -c,
        })

    # Long-only box bounds.
    bounds = [(0.0, 1.0)] * n
    result = minimize(
        fun, x0, jac=jac, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10, "disp": False},
    )

    # Numerical clean-up: round small negatives caused by floating point, then
    # renormalise so the simplex constraint holds exactly.
    w = np.clip(result.x, 0.0, None)
    if w.sum() > 0:
        w = w / w.sum()

    # Final feasibility check on the carbon cap (small slack for FP noise).
    feasible_carbon = True
    if carbon is not None and carbon_limit is not None:
        feasible_carbon = float(w @ np.asarray(carbon, dtype=float)) <= carbon_limit + 1e-6

    return w, bool(result.success and feasible_carbon), str(result.message)


# --------------------------------------------------------------------------- #
# Performance simulation                                                       #
# --------------------------------------------------------------------------- #

def _simulate_year(weights: np.ndarray, rets_oos: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    """Compute monthly portfolio returns over implementation year, letting
    weights drift with realised returns (no rebalancing within the year).

    Formulas in the PDF (Section 2.2):
      R_{p,t+k} = w_{t+k-1}' R_{t+k}
      w_{i,t+k-1} = w_{i,t+k-2} * (1 + R_{i,t+k-1}) / (1 + R_{p,t+k-1})
    """
    w = weights.copy()
    out: dict[pd.Timestamp, float] = {}
    # Iterate over the 12 implementation months. Missing returns are treated
    # as 0 (the firm did not trade that month — its weight simply does not change).
    for date, row in rets_oos.fillna(0.0).iterrows():
        r = row.to_numpy(dtype=float)
        rp = float(w @ r)                # portfolio return this month
        out[date] = rp
        if 1.0 + rp <= 0.0:
            # Portfolio wiped out (e.g. mass delisting). Zero the weights and stop.
            w = np.zeros_like(w)
            break
        # Drift the weights to the start of next month (no rebalancing inside the year).
        w = w * (1.0 + r) / (1.0 + rp)
        w = np.clip(w, 0.0, None)
        if w.sum() > 0:
            w /= w.sum()
    return pd.Series(out), w


def _vw_monthly_rebalanced(cap_m: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Value-weighted benchmark with monthly rebalancing (PDF Section 2.3).

    Formula: R^{vw}_{t+1} = Σ_i (Cap_{i,t} / Σ_j Cap_{j,t}) · R_{i,t+1}.
    The weights at end of month t are applied to the returns of month t+1,
    which is enforced by `cap.shift(1)`.
    """
    cap = cap_m.reindex(index=rets.index, columns=rets.columns)
    # Row-wise normalisation -> weights per date that sum to one across firms.
    w = cap.div(cap.sum(axis=1), axis=0)
    w = w.shift(1)
    # Sum the firm contributions, requiring at least one valid pair (date, firm).
    return (w * rets).sum(axis=1, min_count=1)


# --------------------------------------------------------------------------- #
# Statistics + plotting                                                        #
# --------------------------------------------------------------------------- #

def _summary_stats(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in monthly.columns:
        r = monthly[col].dropna()
        ann_ret = (1.0 + r).prod() ** (12.0 / len(r)) - 1.0 if len(r) else np.nan
        ann_vol = r.std(ddof=0) * np.sqrt(12.0) if len(r) else np.nan
        rows.append({
            "portfolio": col,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan,
            "minimum_monthly_return": r.min() if len(r) else np.nan,
            "maximum_monthly_return": r.max() if len(r) else np.nan,
            "cumulative_return": (1.0 + r).prod() - 1.0 if len(r) else np.nan,
        })
    return pd.DataFrame(rows)


def _plot_cumulative(monthly: pd.DataFrame, cols: list[str], path: Path, title: str) -> None:
    cum = (1.0 + monthly[cols]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    cum.plot(ax=ax, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_metric(annual: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    pivot = annual.pivot(index="year", columns="portfolio", values=metric)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(ax=ax, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Allocation year")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main pipeline                                                                #
# --------------------------------------------------------------------------- #

def run() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    prices, rets, emissions, revenues, cap_y, cap_m, names, countries = _load_inputs()

    # Monthly-rebalanced VW for the performance benchmark (Section 2.3).
    vw_monthly = _vw_monthly_rebalanced(cap_m, rets)
    vw_monthly = vw_monthly.loc[f"{PERF_START}-01-01":f"{PERF_END}-12-31"]

    portfolio_labels = ["vw_drift", "mv", "mv_50", "vw_50", "vw_nz"]
    monthly_series = {p: pd.Series(dtype=float) for p in portfolio_labels}

    annual_rows: list[dict] = []
    weight_rows: list[dict] = []
    top_drivers_rows: list[dict] = []
    cf_vw_2013: float | None = None

    for year in range(YEAR_FIRST, YEAR_LAST + 1):
        # Build the eligible universe + estimated Σ, μ, CI, CF-per-weight, w_vw for this year.
        yi = _build_year_inputs(year, prices, rets, emissions, revenues, cap_y, names, countries)

        # Vector e_i = E_i / Cap_i. Carbon footprint of any portfolio is e · w.
        carbon = yi.cf_per_w.to_numpy()

        # --- Section 2.2 — long-only minimum variance (no carbon cap) ---
        w_mv, mv_ok, mv_msg = _solve_qp(yi.cov, objective="variance")
        cf_mv = float(w_mv @ carbon)

        # --- Section 3.2 — min variance with CF ≤ 0.5 · CF(mv) ---
        w_mv50, mv50_ok, mv50_msg = _solve_qp(
            yi.cov, objective="variance",
            carbon=carbon, carbon_limit=0.5 * cf_mv,
        )

        # Carbon footprint of the VW benchmark this year, recorded once for 2013 (NZ anchor).
        cf_vw = float(yi.w_vw @ carbon)
        if cf_vw_2013 is None:
            cf_vw_2013 = cf_vw

        # --- Section 3.3 — tracking-error portfolio with CF ≤ 0.5 · CF(vw) ---
        w_vw50, vw50_ok, vw50_msg = _solve_qp(
            yi.cov, objective="tracking_error", benchmark=yi.w_vw,
            carbon=carbon, carbon_limit=0.5 * cf_vw,
        )

        # --- Section 4.1 — net-zero tracking-error portfolio ---
        # Cap tightens by 10% every year, anchored on the 2013 VW footprint.
        nz_limit = ((1.0 - THETA_NZ) ** (year - YEAR_FIRST + 1)) * cf_vw_2013
        w_nz, nz_ok, nz_msg = _solve_qp(
            yi.cov, objective="tracking_error", benchmark=yi.w_vw,
            carbon=carbon, carbon_limit=nz_limit,
        )

        portfolios = {
            "vw_drift": (yi.w_vw, True, "benchmark (end-of-year cap weights, drift)", cf_vw),
            "mv": (w_mv, mv_ok, mv_msg, np.nan),
            "mv_50": (w_mv50, mv50_ok, mv50_msg, 0.5 * cf_mv),
            "vw_50": (w_vw50, vw50_ok, vw50_msg, 0.5 * cf_vw),
            "vw_nz": (w_nz, nz_ok, nz_msg, nz_limit),
        }

        # Simulate the implementation year and collect statistics.
        for label, (w, ok, msg, cap_lim) in portfolios.items():
            series, _ = _simulate_year(w, yi.rets_oos)
            monthly_series[label] = pd.concat([monthly_series[label], series])

            cf = float(w @ carbon)
            waci = float(w @ yi.ci.to_numpy())
            annual_rows.append({
                "year": year,
                "implementation_year": year + 1,
                "portfolio": label,
                "n_assets": len(yi.isins),
                "waci_tco2e_per_musd_revenue": waci,
                "carbon_footprint_tco2e_per_musd_invested": cf,
                "attributed_emissions_tco2e_starting_wealth": cf * STARTING_WEALTH_MUSD,
                "carbon_limit": cap_lim,
                "optimization_success": ok,
                "optimization_message": msg,
            })

            active = pd.Series(w, index=yi.isins)
            for isin, weight in active[active > 1e-6].sort_values(ascending=False).items():
                weight_rows.append({
                    "year": year,
                    "portfolio": label,
                    "ISIN": isin,
                    "name": yi.names.get(isin, ""),
                    "country": yi.countries.get(isin, ""),
                    "weight": weight,
                    "ci": yi.ci.get(isin, np.nan),
                    "cf_per_weight": yi.cf_per_w.get(isin, np.nan),
                })

        # Top-10 WACI drivers using the VW weights (PDF Section 3.1 prompt).
        contrib = pd.DataFrame({
            "ISIN": yi.isins,
            "name": yi.names.reindex(yi.isins).to_numpy(),
            "country": yi.countries.reindex(yi.isins).to_numpy(),
            "ci": yi.ci.to_numpy(),
            "vw_weight": yi.w_vw,
            "vw_waci_contribution": yi.w_vw * yi.ci.to_numpy(),
        }).sort_values("vw_waci_contribution", ascending=False)
        for rank, row in enumerate(contrib.head(10).itertuples(index=False), start=1):
            top_drivers_rows.append({"year": year, "rank": rank, **row._asdict()})

        print(
            f"{year}: n={len(yi.isins):>3} | "
            f"CF vw={cf_vw:7.2f} mv={cf_mv:7.2f} mv50={float(w_mv50@carbon):7.2f} "
            f"vw50={float(w_vw50@carbon):7.2f} nz={float(w_nz@carbon):7.2f} "
            f"(nz_limit={nz_limit:.2f})"
        )

    # ----- Assemble panels --------------------------------------------------
    monthly = pd.DataFrame(monthly_series).sort_index()
    monthly = monthly.loc[f"{PERF_START}-01-01":f"{PERF_END}-12-31"]
    monthly["vw_monthly"] = vw_monthly.reindex(monthly.index)
    # `vw` in the report = monthly-rebalanced VW (matches Section 2.3 strictly)
    monthly = monthly.rename(columns={"vw_monthly": "vw"})

    annual = pd.DataFrame(annual_rows)
    weights = pd.DataFrame(weight_rows)
    drivers = pd.DataFrame(top_drivers_rows)
    stats = _summary_stats(monthly[["vw", "vw_drift", "mv", "mv_50", "vw_50", "vw_nz"]])

    # ----- Exclusions / overweights vs VW each year -------------------------
    vw_w = (weights[weights["portfolio"] == "vw_drift"]
            .set_index(["year", "ISIN"])["weight"])
    excl_rows = []
    for (year, label), grp in weights.groupby(["year", "portfolio"]):
        if label == "vw_drift":
            continue
        held = grp.set_index("ISIN")["weight"]
        # Firms in VW but excluded by the constrained portfolio
        vw_year = vw_w.xs(year, level="year") if year in vw_w.index.get_level_values(0) else pd.Series(dtype=float)
        excluded = vw_year[~vw_year.index.isin(held.index)]
        for isin, w in excluded.sort_values(ascending=False).head(10).items():
            excl_rows.append({"year": year, "portfolio": label, "kind": "excluded_vs_vw",
                              "ISIN": isin, "weight_vw": float(w), "weight_p": 0.0})
        # Largest overweights vs VW
        merged = held.to_frame("weight_p").join(vw_year.to_frame("weight_vw"), how="outer").fillna(0.0)
        merged["over"] = merged["weight_p"] - merged["weight_vw"]
        for isin, row in merged.sort_values("over", ascending=False).head(10).iterrows():
            excl_rows.append({"year": year, "portfolio": label, "kind": "overweight_vs_vw",
                              "ISIN": isin, "weight_vw": float(row["weight_vw"]),
                              "weight_p": float(row["weight_p"])})
    exclusions = pd.DataFrame(excl_rows)

    # ----- Persist outputs --------------------------------------------------
    monthly.to_csv(OUTPUTS / "part3_4_monthly_portfolio_returns.csv", index_label="date")
    stats.to_csv(OUTPUTS / "part3_4_performance_summary.csv", index=False)
    annual.to_csv(OUTPUTS / "part3_4_annual_carbon_metrics.csv", index=False)
    weights.to_csv(OUTPUTS / "part3_4_portfolio_weights.csv", index=False)
    drivers.to_csv(OUTPUTS / "part3_4_top10_waci_drivers_by_year.csv", index=False)
    exclusions.to_csv(OUTPUTS / "part3_4_exclusions_overweights.csv", index=False)

    # ----- Figures ----------------------------------------------------------
    _plot_cumulative(monthly, ["vw", "mv", "mv_50"],
                     FIGURES / "part3_mv_vs_mv50_cumulative.png",
                     "Cumulative performance — VW (monthly), MV, MV(0.5)")
    _plot_cumulative(monthly, ["vw", "vw_50"],
                     FIGURES / "part3_vw_vs_vw50_cumulative.png",
                     "Cumulative performance — VW vs VW(0.5)")
    _plot_cumulative(monthly, ["vw", "vw_50", "vw_nz"],
                     FIGURES / "part4_vw_vw50_netzero_cumulative.png",
                     "Cumulative performance — VW, VW(0.5), VW(NZ)")
    _plot_metric(annual, "waci_tco2e_per_musd_revenue",
                 FIGURES / "part3_4_waci_by_portfolio.png",
                 "WACI by portfolio (tCO2e per USD-million revenue)")
    _plot_metric(annual, "carbon_footprint_tco2e_per_musd_invested",
                 FIGURES / "part3_4_carbon_footprint_by_portfolio.png",
                 "Carbon footprint by portfolio (tCO2e per USD-million invested)")

    # Net-zero target path vs realised
    path_df = annual[annual["portfolio"].isin(["vw_drift", "vw_nz"])].copy()
    path_df["target"] = [
        ((1.0 - THETA_NZ) ** (y - YEAR_FIRST + 1)) * cf_vw_2013
        for y in path_df["year"]
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, g in path_df.groupby("portfolio"):
        ax.plot(g["year"], g["carbon_footprint_tco2e_per_musd_invested"],
                marker="o", linewidth=2, label=label)
    ax.plot(path_df["year"].unique(),
            [((1.0 - THETA_NZ) ** (y - YEAR_FIRST + 1)) * cf_vw_2013 for y in sorted(path_df["year"].unique())],
            linestyle="--", color="black", label="net-zero target")
    ax.set_title("Net-zero carbon path vs realised footprint")
    ax.set_xlabel("Allocation year")
    ax.set_ylabel("CF (tCO2e per USD-million invested)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "part4_netzero_path.png", dpi=160)
    plt.close(fig)

    print("\nPerformance summary:")
    print(stats.to_string(index=False))
    print(f"\nSaved Part III-IV outputs to {OUTPUTS}")


if __name__ == "__main__":
    run()

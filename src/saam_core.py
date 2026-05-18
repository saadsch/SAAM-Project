"""Shared helpers for the SAAM project — used by both `src/05_part3.py`
and `src/06_part4.py`.

The functions here implement the PDF rules that are *common* to Parts III
and IV: data loading, price cleaning, eligibility filters, covariance
estimation, the SLSQP-based long-only QP, the drift-based monthly
simulation, the strict monthly-rebalanced VW benchmark, and the headline
statistics (Sharpe with the Pacific risk-free rate).

PDF sections that consume these helpers:
  - Section 1   — universe definition, carbon scope
  - Section 2.1 — price cleaning + eligibility filters
  - Section 2.2 — long-only minimum variance + drift simulation
  - Section 2.3 — monthly-rebalanced value-weighted benchmark
  - Section 3.* — re-used in src/05_part3.py
  - Section 4   — re-used in src/06_part4.py
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

# Annual allocation years (end-of-year decisions); implementation runs the
# following year.
YEAR_FIRST = 2013
YEAR_LAST = 2024
PERF_START = 2014
PERF_END = 2025

ESTIM_MONTHS = 120          # 10 years of monthly returns
MIN_OBS = 36                # >= 3 years available within the window
STALE_THRESHOLD = 0.50      # at most 50% zero returns in the window
STARTING_WEALTH_MUSD = 1.0  # V_2013 = USD 1m (used to attribute emissions)

CARBON_SCOPE = "scope1"     # switch to "scope2" or "scope1_plus_scope2"
THETA_NZ = 0.10             # 10% annual carbon-footprint reduction (Part 4)

RIDGE_EPS = 1e-6            # small ridge added to Sigma for numerical safety

# Pacific monthly risk-free rate (RF in percent per month, indexed YYYYMM).
RF_PATH = PROCESSED / "rf_rate.csv"


# --------------------------------------------------------------------------- #
# PDF Section 2.1 — data loading and cleaning                                  #
# --------------------------------------------------------------------------- #

def _read_wide(path: Path, date_columns: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Read a Datastream-style wide CSV.

    Layout: one row per firm, identified by ISIN; first columns are
    NAME / ISIN, then one column per period (year or month-end date).
    Returns `(df, names)` where `df` is indexed by ISIN with numeric
    period columns, and `names` maps ISIN -> NAME for display.
    """
    df = pd.read_csv(path)
    df = df[df["ISIN"].notna()].copy()
    df["ISIN"] = df["ISIN"].astype(str)
    names = df.set_index("ISIN")["NAME"].astype(str) if "NAME" in df.columns else pd.Series(dtype=str)
    df = df.set_index("ISIN").drop(columns=["NAME"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df.columns = pd.to_datetime(df.columns).normalize() if date_columns else df.columns.astype(int)
    return df, names


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """PDF Section 2.1 price-cleaning rules.

    - prices < 0.5 -> treated as missing,
    - middle missing values -> forward fill,
    - trailing missing values (delisting) -> set to 0 so the realised
      monthly return on the delisting month equals -100%.
    """
    prices = prices.T.sort_index()
    prices[prices < 0.5] = np.nan
    has = prices.notna().astype(np.int8)
    ahead = has.iloc[::-1].cumsum(axis=0).iloc[::-1]
    trailing = prices.isna() & (ahead == 0)
    prices = prices.ffill(axis=0)
    prices[trailing] = 0.0
    return prices


def _returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly returns with delisting handled as -100%."""
    prev = prices.shift(1)
    rets = prices / prev.replace(0.0, np.nan) - 1.0
    rets[(prices == 0.0) & (prev > 0.0)] = -1.0
    return rets.replace([np.inf, -np.inf], np.nan)


def _annual_panel(path: Path) -> pd.DataFrame:
    """Annual wide panel forward-filled across years (PDF rule on missing
    carbon/revenue entries)."""
    df, _ = _read_wide(path, date_columns=False)
    return df.sort_index(axis=1).ffill(axis=1)


def _load_rf(path: Path = RF_PATH) -> pd.Series:
    """Pacific monthly risk-free rate.

    File layout: first column = YYYYMM integer, second column = RF in
    percent per month (same convention as `src/MVP-construction.ipynb`).
    Returned as a Series of monthly RF in fraction (RF/100), indexed by
    month-end timestamps.
    """
    if not path.exists():
        return pd.Series(dtype=float)
    raw = pd.read_csv(path)
    if raw.shape[1] < 2:
        return pd.Series(dtype=float)
    raw.columns = ["period", "rf"]
    raw = raw.dropna()
    period_int = raw["period"].astype(int).astype(str).str.zfill(6)
    dates = pd.to_datetime(period_int, format="%Y%m") + pd.offsets.MonthEnd(0)
    rf = pd.Series(raw["rf"].astype(float).to_numpy() / 100.0, index=dates).sort_index()
    rf.name = "rf"
    return rf


def load_inputs():
    """Load every panel used by Parts III & IV and align them on the
    intersection of ISINs.

    Returns: prices, rets, emissions, revenues, cap_y, cap_m, names, countries
    """
    prices_raw, _ = _read_wide(PROCESSED / "Clean_Prices_Pacific.csv", date_columns=True)
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
# PDF Section 1 + 2.1 — eligible universe at end-Y                             #
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
    ci: pd.Series                 # carbon intensity tCO2e per USD-million revenue
    cf_per_w: pd.Series           # E_i / Cap_i (contribution to footprint per unit weight)
    w_vw: np.ndarray              # end-of-year value weights


def build_year_inputs(year, prices, rets, emissions, revenues, cap_y, names, countries) -> YearInputs:
    """Build the eligible universe and the optimisation inputs for one
    allocation year (end of `year`). Implementation runs over `year + 1`.
    """
    est_start = pd.Timestamp(year - 9, 1, 1)
    est_end = pd.Timestamp(year, 12, 31)
    oos_start = pd.Timestamp(year + 1, 1, 1)
    oos_end = pd.Timestamp(year + 1, 12, 31)

    rets_window_all = rets.loc[est_start:est_end]
    rets_oos_all = rets.loc[oos_start:oos_end]

    obs = rets_window_all.count()
    zero_ratio = (rets_window_all == 0.0).sum() / obs.replace(0, np.nan)
    last_price_valid = prices.loc[:est_end].iloc[-1].gt(0)

    # Eligibility: PDF Sections 1 + 2.1 — carbon, revenue, market cap
    # available at end-Y; price > 0 at end-Y; >= 36 monthly observations;
    # <= 50% zero returns in the 120-month window.
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

    rets_est = rets_window_all[isins].iloc[-ESTIM_MONTHS:]
    rets_oos = rets_oos_all[isins]

    # Covariance: complete-case rows (fall back to mean-fill when <36 complete).
    complete = rets_est.dropna(how="any")
    if len(complete) < MIN_OBS:
        filled = rets_est.apply(lambda s: s.fillna(s.mean()), axis=0).fillna(0.0)
        mu = filled.mean().to_numpy()
        cov = np.cov(filled.to_numpy(), rowvar=False, ddof=0)
    else:
        mu = complete.mean().to_numpy()
        cov = np.cov(complete.to_numpy(), rowvar=False, ddof=0)
    cov = (cov + cov.T) / 2.0
    cov += RIDGE_EPS * np.eye(cov.shape[0])

    cap_yi = cap_y.loc[isins, year].astype(float)
    emi_yi = emissions.loc[isins, year].astype(float)
    rev_yi = revenues.loc[isins, year].astype(float)

    # PDF formulas:
    #   CI_i      = E_i / (Rev_i / 1000)   -> tCO2e per USD-million revenue
    #   E_i/Cap_i = contribution to CF per unit weight (cap in M$ -> tCO2e per M$)
    ci = emi_yi / (rev_yi / 1000.0)
    cf_per_w = emi_yi / cap_yi
    w_vw = (cap_yi / cap_yi.sum()).to_numpy()

    return YearInputs(
        year=year, isins=isins,
        names=names.reindex(isins),
        countries=countries.reindex(isins),
        rets_est=rets_est, rets_oos=rets_oos,
        mu=mu, cov=cov,
        cap_y=cap_yi, emissions=emi_yi, revenues=rev_yi,
        ci=ci, cf_per_w=cf_per_w, w_vw=w_vw,
    )


# --------------------------------------------------------------------------- #
# PDF Sections 2.2 / 3.2 / 3.3 / 4 — long-only QP via SLSQP                     #
# --------------------------------------------------------------------------- #

def solve_qp(
    cov: np.ndarray,
    objective: str = "variance",
    benchmark: np.ndarray | None = None,
    carbon: np.ndarray | None = None,
    carbon_limit: float | None = None,
) -> tuple[np.ndarray, bool, str]:
    """Long-only constrained QP, solved with `scipy.optimize.minimize` /
    SLSQP and analytic gradients.

    objective="variance"        -> min  w' Σ w                (Section 2.2 / 3.2)
    objective="tracking_error"  -> min (w - benchmark)' Σ (w - benchmark)
                                                              (Section 3.3 / 4)
    subject to:
        Σ w_i = 1
        0 <= w_i <= 1
        carbon · w <= carbon_limit   (when provided)
    """
    n = cov.shape[0]
    if benchmark is None:
        benchmark = np.full(n, 1.0 / n)

    # Warm start.
    if objective == "tracking_error":
        x0 = np.asarray(benchmark, dtype=float).copy()
    else:
        x0 = np.full(n, 1.0 / n)

    # If the warm start violates the cap, mix it with the lowest-carbon vertex.
    if carbon is not None and carbon_limit is not None:
        if float(x0 @ carbon) > carbon_limit:
            lo_i = int(np.argmin(carbon))
            unit = np.zeros(n)
            unit[lo_i] = 1.0
            base = float(x0 @ carbon)
            low = float(unit @ carbon)
            if low > carbon_limit:
                x0 = unit
            else:
                alpha = (base - carbon_limit) / max(base - low, 1e-12)
                alpha = float(np.clip(alpha + 1e-3, 0.0, 1.0))
                x0 = (1.0 - alpha) * x0 + alpha * unit

    if objective == "variance":
        def fun(w): return float(w @ cov @ w)
        def jac(w): return 2.0 * cov @ w
    elif objective == "tracking_error":
        b = np.asarray(benchmark, dtype=float)
        def fun(w):
            d = w - b
            return float(d @ cov @ d)
        def jac(w): return 2.0 * cov @ (w - b)
    else:
        raise ValueError(objective)

    constraints = [{"type": "eq",
                    "fun": lambda w: float(np.sum(w) - 1.0),
                    "jac": lambda w: np.ones_like(w)}]
    if carbon is not None and carbon_limit is not None:
        c = np.asarray(carbon, dtype=float)
        constraints.append({
            "type": "ineq",
            "fun": lambda w, c=c, L=float(carbon_limit): float(L - w @ c),
            "jac": lambda w, c=c: -c,
        })

    bounds = [(0.0, 1.0)] * n
    result = minimize(
        fun, x0, jac=jac, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10, "disp": False},
    )

    w = np.clip(result.x, 0.0, None)
    if w.sum() > 0:
        w = w / w.sum()

    feasible_carbon = True
    if carbon is not None and carbon_limit is not None:
        feasible_carbon = float(w @ np.asarray(carbon, dtype=float)) <= carbon_limit + 1e-6

    return w, bool(result.success and feasible_carbon), str(result.message)


# --------------------------------------------------------------------------- #
# PDF Section 2.2 / 2.3 — simulation and benchmark                             #
# --------------------------------------------------------------------------- #

def simulate_year(weights: np.ndarray, rets_oos: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    """Monthly portfolio returns over the implementation year with weights
    drifting per PDF Section 2.2 (no rebalancing within the year).
    """
    w = weights.copy()
    out: dict[pd.Timestamp, float] = {}
    for date, row in rets_oos.fillna(0.0).iterrows():
        r = row.to_numpy(dtype=float)
        rp = float(w @ r)
        out[date] = rp
        if 1.0 + rp <= 0.0:
            w = np.zeros_like(w)
            break
        w = w * (1.0 + r) / (1.0 + rp)
        w = np.clip(w, 0.0, None)
        if w.sum() > 0:
            w /= w.sum()
    return pd.Series(out), w


def vw_monthly_rebalanced(cap_m: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Strict PDF Section 2.3 VW benchmark: R^{vw}_{t+1} = Σ_i (Cap_{i,t} /
    Σ_j Cap_{j,t}) · R_{i,t+1}. The shift(1) enforces previous-period
    weights -> next-period returns."""
    cap = cap_m.reindex(index=rets.index, columns=rets.columns)
    w = cap.div(cap.sum(axis=1), axis=0)
    w = w.shift(1)
    return (w * rets).sum(axis=1, min_count=1)


# --------------------------------------------------------------------------- #
# Statistics + plotting                                                        #
# --------------------------------------------------------------------------- #

def summary_stats(
    monthly: pd.DataFrame,
    rf: pd.Series | None = None,
    benchmark: str = "vw",
) -> pd.DataFrame:
    """Per-portfolio headline statistics with an RF-aware Sharpe.

    Sharpe = (annualised return - annualised RF) / annualised volatility,
    where annualised RF = monthly_rf.mean() * 12 (rf in fraction). The
    `tracking_error_vs_vw` column is the annualised std of `R_p - R_bench`.
    """
    rows: list[dict] = []
    bench_series = monthly[benchmark] if benchmark in monthly.columns else None
    for col in monthly.columns:
        r = monthly[col].dropna()
        ann_ret = (1.0 + r).prod() ** (12.0 / len(r)) - 1.0 if len(r) else np.nan
        ann_vol = r.std(ddof=0) * np.sqrt(12.0) if len(r) else np.nan
        if rf is not None and len(r):
            rf_aligned = rf.reindex(r.index).dropna()
            ann_rf = float(rf_aligned.mean() * 12.0) if len(rf_aligned) else 0.0
        else:
            ann_rf = 0.0
        sharpe = (ann_ret - ann_rf) / ann_vol if ann_vol and ann_vol > 0 else np.nan
        if bench_series is not None and col != benchmark and len(r):
            diff = (monthly[col] - bench_series).dropna()
            te = float(diff.std(ddof=0) * np.sqrt(12.0)) if len(diff) else np.nan
        else:
            te = np.nan
        rows.append({
            "portfolio": col,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "annualized_risk_free": ann_rf,
            "sharpe_ratio": sharpe,
            "tracking_error_vs_vw": te,
            "minimum_monthly_return": r.min() if len(r) else np.nan,
            "maximum_monthly_return": r.max() if len(r) else np.nan,
            "cumulative_return": (1.0 + r).prod() - 1.0 if len(r) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_cumulative(monthly: pd.DataFrame, cols: list[str], path: Path, title: str) -> None:
    """Cumulative growth-of-$1 plot. We prepend a row of 1.0 one month
    before the first return so the curve visually starts at $1."""
    cum = (1.0 + monthly[cols]).cumprod()
    if len(cum):
        first_date = cum.index.min() - pd.offsets.MonthEnd(1)
        start = pd.DataFrame(1.0, index=[first_date], columns=cols)
        cum = pd.concat([start, cum])
    fig, ax = plt.subplots(figsize=(10, 6))
    cum.plot(ax=ax, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric(annual: pd.DataFrame, metric: str, path: Path, title: str) -> None:
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

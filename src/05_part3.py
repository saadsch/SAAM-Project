"""SAAM Project - Part III: 50% carbon-footprint reduction.

Implements PDF Sections 3.1, 3.2, 3.3 and 3.4 on the Pacific region.

Pipeline (each Section is clearly banner-tagged below):
  - Section 2.3  monthly-rebalanced VW benchmark (consumed as P_vw input)
  - Section 2.2  long-only minimum variance P_mv_oos
  - Section 3.1  carbon metrics (CI, WACI, CF) for P_mv_oos and P_vw
  - Section 3.2  P_mv_oos(0.5): min variance with CF <= 0.5 * CF(P_mv_oos)
  - Section 3.3  P_vw_oos(0.5): TE minimisation with CF <= 0.5 * CF(P_vw)
  - Section 3.4  feasibility / TE / cumulative comparison plots

Part III deliverables (all paths relative to the repo root):

3.1
  outputs/tables/part3_carbon_metrics_mv_vw.csv
  outputs/figures/part3_waci_mv_vs_vw.png
  outputs/figures/part3_carbon_footprint_mv_vs_vw.png
  outputs/tables/part3_top10_waci_contributors.csv
  outputs/tables/part3_top10_cf_contributors.csv

3.2
  outputs/tables/part3_returns_mv_carbon50.csv
  outputs/tables/part3_summary_mv_vs_mv_carbon50.csv
  outputs/tables/part3_weights_mv_carbon50.csv
  outputs/tables/part3_constraint_slack_mv_carbon50.csv
  outputs/figures/part3_cumulative_mv_vs_mv_carbon50.png
  outputs/figures/part3_cf_mv_vs_mv_carbon50.png
  outputs/figures/part3_waci_mv_vs_mv_carbon50.png

3.3
  outputs/tables/part3_returns_vw_carbon50.csv
  outputs/tables/part3_summary_vw_vs_vw_carbon50.csv
  outputs/tables/part3_weights_vw_carbon50.csv
  outputs/tables/part3_tracking_error_vw_carbon50.csv
  outputs/tables/part3_constraint_slack_vw_carbon50.csv
  outputs/figures/part3_cumulative_vw_vs_vw_carbon50.png
  outputs/figures/part3_cf_vw_vs_vw_carbon50.png
  outputs/figures/part3_waci_vw_vs_vw_carbon50.png
  outputs/figures/part3_tracking_error_vw_carbon50.png

Shared helpers live in ``saam_core``. Part IV (Net-Zero) is in 06_part4.py.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from saam_core import (
    FIGURES,
    OUTPUTS,
    PERF_END,
    PERF_START,
    RF_PATH,
    YEAR_FIRST,
    YEAR_LAST,
    build_year_inputs,
    load_inputs,
    plot_cumulative,
    simulate_year,
    solve_qp,
    summary_stats,
    vw_monthly_rebalanced,
    _load_rf,
)

TABLES = OUTPUTS / "tables"


# --------------------------------------------------------------------------- #
# Helper plotters                                                             #
# --------------------------------------------------------------------------- #
def _plot_annual_metric(annual: pd.DataFrame, metric: str, portfolios: list[str],
                        path, title: str, ylabel: str) -> None:
    """Annual line plot of `metric` for the subset of `portfolios`."""
    sub = annual[annual["portfolio"].isin(portfolios)]
    pivot = sub.pivot(index="year", columns="portfolio", values=metric)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot[portfolios].plot(ax=ax, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Allocation year")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)

    # ===================================================================== #
    # === Section 2.1 - Load inputs + clean prices + eligibility filters == #
    # ===================================================================== #
    prices, rets, emissions, revenues, cap_y, cap_m, names, countries = load_inputs()

    # ===================================================================== #
    # === Section 2.3 - Value-weighted benchmark (monthly rebalanced)    == #
    # ===================================================================== #
    vw_monthly = vw_monthly_rebalanced(cap_m, rets)
    vw_monthly = vw_monthly.loc[f"{PERF_START}-01-01":f"{PERF_END}-12-31"]

    # Containers
    portfolio_labels = ["vw_drift", "mv", "mv_50", "vw_50"]
    monthly_series = {p: pd.Series(dtype=float) for p in portfolio_labels}

    annual_rows: list[dict] = []
    weight_rows: list[dict] = []
    waci_driver_rows: list[dict] = []
    cf_driver_rows: list[dict] = []
    te_rows: list[dict] = []
    slack_rows: list[dict] = []

    # ===================================================================== #
    # === Annual loop  Y = 2013 .. 2024  (implementation year Y+1)        == #
    # ===================================================================== #
    for year in range(YEAR_FIRST, YEAR_LAST + 1):
        yi = build_year_inputs(
            year, prices, rets, emissions, revenues, cap_y, names, countries
        )
        carbon = yi.cf_per_w.to_numpy()  # e_i = E_i / Cap_i  (tCO2e per M$)

        # --- Section 2.2 - P_mv_oos: long-only minimum variance --------- #
        w_mv, mv_ok, mv_msg = solve_qp(yi.cov, objective="variance")
        cf_mv = float(w_mv @ carbon)

        # --- Section 3.2 - P_mv_oos(0.5): MV s.t. CF <= 0.5 * CF(mv) ---- #
        # 50%-cap benchmark is P_mv_oos itself (NOT the VW portfolio).
        w_mv50, mv50_ok, mv50_msg = solve_qp(
            yi.cov, objective="variance",
            carbon=carbon, carbon_limit=0.5 * cf_mv,
        )

        # --- Section 3.3 - P_vw_oos(0.5): min TE s.t. CF <= 0.5 * CF(vw)  #
        cf_vw = float(yi.w_vw @ carbon)
        w_vw50, vw50_ok, vw50_msg = solve_qp(
            yi.cov, objective="tracking_error", benchmark=yi.w_vw,
            carbon=carbon, carbon_limit=0.5 * cf_vw,
        )

        portfolios = {
            "vw_drift": (yi.w_vw, True, "benchmark (drift)", cf_vw),
            "mv":       (w_mv,    mv_ok,    mv_msg,    np.nan),
            "mv_50":    (w_mv50,  mv50_ok,  mv50_msg,  0.5 * cf_mv),
            "vw_50":    (w_vw50,  vw50_ok,  vw50_msg,  0.5 * cf_vw),
        }

        # --- Section 2.2 - simulate implementation year (drift) ---------- #
        bench_w = yi.w_vw
        bench_series_year = None
        for label, (w, ok, msg, cap_lim) in portfolios.items():
            series, _ = simulate_year(w, yi.rets_oos)
            monthly_series[label] = pd.concat([monthly_series[label], series])
            if label == "vw_drift":
                bench_series_year = series

            cf = float(w @ carbon)
            waci = float(w @ yi.ci.to_numpy())

            # --- Section 3.1 - carbon metrics row ------------------------ #
            annual_rows.append({
                "year": year,
                "implementation_year": year + 1,
                "portfolio": label,
                "n_assets": len(yi.isins),
                "waci_tco2e_per_musd_revenue": waci,
                "carbon_footprint_tco2e_per_musd_invested": cf,
                "carbon_limit": cap_lim,
                "optimization_success": bool(ok),
                "optimization_message": msg,
            })

            # --- Section 3.* - constraint slack row --------------------- #
            cap_value = (cap_lim
                         if not (isinstance(cap_lim, float) and np.isnan(cap_lim))
                         else None)
            slack = (cap_value - cf) if cap_value is not None else np.nan
            slack_rows.append({
                "year": year,
                "implementation_year": year + 1,
                "portfolio": label,
                "carbon_footprint": cf,
                "carbon_limit": cap_value if cap_value is not None else np.nan,
                "slack": slack,
                "satisfied_within_1e-6": bool(
                    cap_value is None or slack >= -1e-6
                ),
                "optimization_success": bool(ok),
                "optimization_message": msg,
            })

            # --- weights table ----------------------------------------- #
            active = pd.Series(w, index=yi.isins)
            for isin, weight in active[active > 1e-6].sort_values(ascending=False).items():
                weight_rows.append({
                    "year": year,
                    "portfolio": label,
                    "ISIN": isin,
                    "name": yi.names.get(isin, ""),
                    "country": yi.countries.get(isin, ""),
                    "weight": float(weight),
                    "ci": float(yi.ci.get(isin, np.nan)),
                    "cf_per_weight": float(yi.cf_per_w.get(isin, np.nan)),
                })

            # --- Section 3.3 / 3.4 - tracking error vs VW --------------- #
            diff_w = w - bench_w
            ex_ante_te = float(np.sqrt(max(diff_w @ yi.cov @ diff_w, 0.0) * 12.0))
            if bench_series_year is not None and label != "vw_drift":
                aligned = series.reindex(bench_series_year.index)
                rel = (aligned - bench_series_year).dropna()
                ex_post_te = float(rel.std(ddof=0) * np.sqrt(12.0)) if len(rel) > 1 else np.nan
            else:
                ex_post_te = np.nan
            te_rows.append({
                "year": year,
                "implementation_year": year + 1,
                "portfolio": label,
                "ex_ante_tracking_error_annual": ex_ante_te,
                "ex_post_tracking_error_annual": ex_post_te,
            })

        # --- Section 3.1 - top-10 WACI drivers (using VW weights) -------- #
        waci_contrib = pd.DataFrame({
            "ISIN": yi.isins,
            "name": yi.names.reindex(yi.isins).to_numpy(),
            "country": yi.countries.reindex(yi.isins).to_numpy(),
            "ci_tco2e_per_musd_revenue": yi.ci.to_numpy(),
            "vw_weight": yi.w_vw,
            "vw_waci_contribution": yi.w_vw * yi.ci.to_numpy(),
        }).sort_values("vw_waci_contribution", ascending=False)
        for rank, row in enumerate(waci_contrib.head(10).itertuples(index=False), start=1):
            waci_driver_rows.append({"year": year, "rank": rank, **row._asdict()})

        # --- Section 3.1 - top-10 CF drivers (using VW weights) ---------- #
        cf_contrib = pd.DataFrame({
            "ISIN": yi.isins,
            "name": yi.names.reindex(yi.isins).to_numpy(),
            "country": yi.countries.reindex(yi.isins).to_numpy(),
            "cf_per_weight_tco2e_per_musd_invested": yi.cf_per_w.to_numpy(),
            "vw_weight": yi.w_vw,
            "vw_cf_contribution": yi.w_vw * yi.cf_per_w.to_numpy(),
        }).sort_values("vw_cf_contribution", ascending=False)
        for rank, row in enumerate(cf_contrib.head(10).itertuples(index=False), start=1):
            cf_driver_rows.append({"year": year, "rank": rank, **row._asdict()})

        print(
            f"{year}: n={len(yi.isins):>3} | "
            f"CF vw={cf_vw:7.2f}  mv={cf_mv:7.2f}  "
            f"mv50={float(w_mv50@carbon):7.2f}  vw50={float(w_vw50@carbon):7.2f}"
        )

    # ===================================================================== #
    # === Assemble monthly panel                                          == #
    # ===================================================================== #
    monthly = pd.DataFrame(monthly_series).sort_index()
    monthly = monthly.loc[f"{PERF_START}-01-01":f"{PERF_END}-12-31"]
    monthly["vw"] = vw_monthly.reindex(monthly.index)
    cols_order = ["vw", "vw_drift", "mv", "mv_50", "vw_50"]
    monthly = monthly[cols_order]

    annual = pd.DataFrame(annual_rows)
    weights = pd.DataFrame(weight_rows)
    waci_drivers = pd.DataFrame(waci_driver_rows)
    cf_drivers = pd.DataFrame(cf_driver_rows)
    te_table = pd.DataFrame(te_rows)
    slack_table = pd.DataFrame(slack_rows)

    rf_monthly = _load_rf(RF_PATH)
    stats = summary_stats(monthly, rf=rf_monthly, benchmark="vw")

    # ===================================================================== #
    # === Section 3.1 - tables and figures (P_mv_oos vs P_vw)            == #
    # ===================================================================== #
    metrics_3_1 = annual[annual["portfolio"].isin(["vw_drift", "mv"])].copy()
    metrics_3_1["portfolio"] = metrics_3_1["portfolio"].replace({"vw_drift": "vw"})
    metrics_3_1.to_csv(TABLES / "part3_carbon_metrics_mv_vw.csv", index=False)

    waci_drivers.to_csv(TABLES / "part3_top10_waci_contributors.csv", index=False)
    cf_drivers.to_csv(TABLES / "part3_top10_cf_contributors.csv", index=False)

    _plot_annual_metric(
        metrics_3_1, "waci_tco2e_per_musd_revenue", ["vw", "mv"],
        FIGURES / "part3_waci_mv_vs_vw.png",
        "Section 3.1 - WACI: P_mv_oos vs P_vw",
        "WACI (tCO2e per USD-million revenue)",
    )
    _plot_annual_metric(
        metrics_3_1, "carbon_footprint_tco2e_per_musd_invested", ["vw", "mv"],
        FIGURES / "part3_carbon_footprint_mv_vs_vw.png",
        "Section 3.1 - Carbon footprint: P_mv_oos vs P_vw",
        "CF (tCO2e per USD-million invested)",
    )

    # ===================================================================== #
    # === Section 3.2 - P_mv_oos vs P_mv_oos(0.5)                        == #
    # ===================================================================== #
    monthly[["mv", "mv_50"]].to_csv(
        TABLES / "part3_returns_mv_carbon50.csv", index_label="date"
    )
    stats[stats["portfolio"].isin(["mv", "mv_50"])].to_csv(
        TABLES / "part3_summary_mv_vs_mv_carbon50.csv", index=False
    )
    weights[weights["portfolio"] == "mv_50"].to_csv(
        TABLES / "part3_weights_mv_carbon50.csv", index=False
    )
    slack_table[slack_table["portfolio"] == "mv_50"].to_csv(
        TABLES / "part3_constraint_slack_mv_carbon50.csv", index=False
    )

    plot_cumulative(
        monthly, ["mv", "mv_50"],
        FIGURES / "part3_cumulative_mv_vs_mv_carbon50.png",
        "Section 3.2 - Cumulative growth of $1: MV vs MV(carbon -50%)",
    )
    _plot_annual_metric(
        annual, "carbon_footprint_tco2e_per_musd_invested", ["mv", "mv_50"],
        FIGURES / "part3_cf_mv_vs_mv_carbon50.png",
        "Section 3.2 - Carbon footprint: MV vs MV(carbon -50%)",
        "CF (tCO2e per USD-million invested)",
    )
    _plot_annual_metric(
        annual, "waci_tco2e_per_musd_revenue", ["mv", "mv_50"],
        FIGURES / "part3_waci_mv_vs_mv_carbon50.png",
        "Section 3.2 - WACI: MV vs MV(carbon -50%)",
        "WACI (tCO2e per USD-million revenue)",
    )

    # ===================================================================== #
    # === Section 3.3 - P_vw vs P_vw_oos(0.5)                            == #
    # ===================================================================== #
    monthly[["vw", "vw_50"]].to_csv(
        TABLES / "part3_returns_vw_carbon50.csv", index_label="date"
    )
    stats[stats["portfolio"].isin(["vw", "vw_50"])].to_csv(
        TABLES / "part3_summary_vw_vs_vw_carbon50.csv", index=False
    )
    weights[weights["portfolio"] == "vw_50"].to_csv(
        TABLES / "part3_weights_vw_carbon50.csv", index=False
    )
    te_table[te_table["portfolio"] == "vw_50"].to_csv(
        TABLES / "part3_tracking_error_vw_carbon50.csv", index=False
    )
    slack_table[slack_table["portfolio"] == "vw_50"].to_csv(
        TABLES / "part3_constraint_slack_vw_carbon50.csv", index=False
    )

    plot_cumulative(
        monthly, ["vw", "vw_50"],
        FIGURES / "part3_cumulative_vw_vs_vw_carbon50.png",
        "Section 3.3 - Cumulative growth of $1: VW vs VW(carbon -50%)",
    )
    annual_vw_view = annual.copy()
    annual_vw_view["portfolio"] = annual_vw_view["portfolio"].replace({"vw_drift": "vw"})
    _plot_annual_metric(
        annual_vw_view, "carbon_footprint_tco2e_per_musd_invested", ["vw", "vw_50"],
        FIGURES / "part3_cf_vw_vs_vw_carbon50.png",
        "Section 3.3 - Carbon footprint: VW vs VW(carbon -50%)",
        "CF (tCO2e per USD-million invested)",
    )
    _plot_annual_metric(
        annual_vw_view, "waci_tco2e_per_musd_revenue", ["vw", "vw_50"],
        FIGURES / "part3_waci_vw_vs_vw_carbon50.png",
        "Section 3.3 - WACI: VW vs VW(carbon -50%)",
        "WACI (tCO2e per USD-million revenue)",
    )

    # Section 3.3 - annualised ex-ante TE
    te_vw50 = te_table[te_table["portfolio"] == "vw_50"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(te_vw50["year"], te_vw50["ex_ante_tracking_error_annual"],
            marker="o", linewidth=2, label="ex-ante")
    ax.plot(te_vw50["year"], te_vw50["ex_post_tracking_error_annual"],
            marker="s", linewidth=2, linestyle="--", label="ex-post")
    ax.set_title("Section 3.3 - Annualised tracking error of VW(carbon -50%) vs VW")
    ax.set_xlabel("Allocation year")
    ax.set_ylabel("Tracking error (annualised)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "part3_tracking_error_vw_carbon50.png", dpi=160)
    plt.close(fig)

    # ===================================================================== #
    # === Section 3.4 - feasibility / status console summary             == #
    # ===================================================================== #
    constrained = slack_table[slack_table["carbon_limit"].notna()
                              & (slack_table["portfolio"] != "vw_drift")]
    if len(constrained):
        worst = constrained["slack"].min()
        if worst < -1e-6:
            raise RuntimeError(f"Part 3 carbon constraint violated: min slack = {worst:.6f}")
        print(f"Part 3 carbon-constraint check OK: min slack = {worst:.6f} tCO2e/M$")
    failed = slack_table[~slack_table["optimization_success"]]
    if len(failed):
        print("WARNING: Part 3 optimizer failed for:")
        print(failed[["year", "portfolio", "optimization_message"]].to_string(index=False))

    # Caches required by Part IV: vw_50 monthly series + carbon metrics +
    # weights so that 06_part4.py can build the joint VW / VW(0.5) / VW(NZ)
    # outputs without re-running the QPs. Stored under `outputs/` (not
    # `outputs/tables/`) so they remain internal artefacts.
    monthly[["vw", "vw_50"]].to_csv(OUTPUTS / "_cache_part3_vw50_monthly.csv",
                                    index_label="date")
    annual_vw_view[annual_vw_view["portfolio"].isin(["vw", "vw_50"])].to_csv(
        OUTPUTS / "_cache_part3_vw_vw50_annual.csv", index=False
    )

    print("\nPart 3 performance summary:")
    print(stats.to_string(index=False))
    print(f"\nSaved Part III deliverables to {OUTPUTS / 'tables'} and {FIGURES}")


if __name__ == "__main__":
    run()

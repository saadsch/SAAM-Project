"""SAAM Project - Part IV: Net-Zero trajectory.

Implements PDF Section 4 on the Pacific region.

Pipeline (each Section is clearly banner-tagged below):
  - Section 2.3  monthly-rebalanced VW benchmark
  - Section 4.1  anchor footprint CF(P_vw)_{Y0}, Y0 = 2013, and dynamic cap
                 C_Y = (1 - theta)^(Y - Y0 + 1) * CF(P_vw)_{Y0}, theta = 10%
  - Section 4.2  P_vw_oos(NZ): min (w - w_vw)' Sigma (w - w_vw)
                 s.t. sum(w) = 1, w >= 0, e' w <= C_Y

Joint VW / VW(0.5) / VW(NZ) comparison plots and the summary use the
VW(0.5) artefacts produced by `05_part3.py` (read from
`outputs/_cache_part3_*`). Run Part 3 before Part 4.

Part IV deliverables (paths relative to the repo root):

  outputs/tables/part4_returns_vw_netzero.csv
  outputs/tables/part4_summary_vw_vs_carbon50_vs_netzero.csv
  outputs/tables/part4_weights_vw_netzero.csv
  outputs/tables/part4_netzero_target_vs_realized_cf.csv
  outputs/tables/part4_tracking_error_vw_netzero.csv
  outputs/tables/part4_constraint_slack_vw_netzero.csv
  outputs/figures/part4_cumulative_vw_vs_carbon50_vs_netzero.png
  outputs/figures/part4_cf_target_vs_realized.png
  outputs/figures/part4_waci_vw_vs_carbon50_vs_netzero.png
  outputs/figures/part4_tracking_error_vw_netzero.png
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
    THETA_NZ,
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


def run() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)

    # ===================================================================== #
    # === Section 2.1 - Load inputs                                       == #
    # ===================================================================== #
    prices, rets, emissions, revenues, cap_y, cap_m, names, countries = load_inputs()

    # ===================================================================== #
    # === Section 2.3 - VW benchmark (monthly rebalanced)                 == #
    # ===================================================================== #
    vw_monthly = vw_monthly_rebalanced(cap_m, rets)
    vw_monthly = vw_monthly.loc[f"{PERF_START}-01-01":f"{PERF_END}-12-31"]

    # ===================================================================== #
    # === Section 4.1 - Anchor: CF(P_vw)_{Y0=2013}                        == #
    # ===================================================================== #
    anchor_year = YEAR_FIRST
    yi_anchor = build_year_inputs(
        anchor_year, prices, rets, emissions, revenues, cap_y, names, countries
    )
    cf_vw_anchor = float(yi_anchor.w_vw @ yi_anchor.cf_per_w.to_numpy())
    print(f"Section 4.1 anchor CF(P_vw)_{anchor_year} = {cf_vw_anchor:.4f} tCO2e/M$")

    monthly_series = {"vw_drift": pd.Series(dtype=float),
                      "vw_nz":    pd.Series(dtype=float)}

    annual_rows: list[dict] = []
    weight_rows: list[dict] = []
    te_rows: list[dict] = []
    slack_rows: list[dict] = []
    path_rows: list[dict] = []

    # ===================================================================== #
    # === Section 4.2 - Annual NZ optimisation  Y = 2013 .. 2024          == #
    # ===================================================================== #
    for year in range(YEAR_FIRST, YEAR_LAST + 1):
        yi = build_year_inputs(
            year, prices, rets, emissions, revenues, cap_y, names, countries
        )
        carbon = yi.cf_per_w.to_numpy()
        cf_vw = float(yi.w_vw @ carbon)

        # --- Section 4.1 - cap for allocation year Y -------------------- #
        cap_nz = ((1.0 - THETA_NZ) ** (year - anchor_year + 1)) * cf_vw_anchor

        # --- Section 4.2 - solve NZ tracking-error QP ------------------- #
        w_nz, nz_ok, nz_msg = solve_qp(
            yi.cov, objective="tracking_error", benchmark=yi.w_vw,
            carbon=carbon, carbon_limit=cap_nz,
        )

        portfolios = {
            "vw_drift": (yi.w_vw, True,  "benchmark (drift)", cf_vw),
            "vw_nz":    (w_nz,    nz_ok, nz_msg,              cap_nz),
        }

        bench_w = yi.w_vw
        bench_series_year = None
        for label, (w, ok, msg, cap_lim) in portfolios.items():
            series, _ = simulate_year(w, yi.rets_oos)
            monthly_series[label] = pd.concat([monthly_series[label], series])
            if label == "vw_drift":
                bench_series_year = series

            cf = float(w @ carbon)
            waci = float(w @ yi.ci.to_numpy())

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
            slack_rows.append({
                "year": year,
                "implementation_year": year + 1,
                "portfolio": label,
                "carbon_footprint": cf,
                "carbon_limit": cap_lim,
                "slack": cap_lim - cf,
                "satisfied_within_1e-6": bool((cap_lim - cf) >= -1e-6),
                "optimization_success": bool(ok),
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
                    "weight": float(weight),
                    "ci": float(yi.ci.get(isin, np.nan)),
                    "cf_per_weight": float(yi.cf_per_w.get(isin, np.nan)),
                })

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

        # --- Section 4.1 - cap-path table row --------------------------- #
        path_rows.append({
            "year": year,
            "implementation_year": year + 1,
            "anchor_cf_vw_2013": cf_vw_anchor,
            "theta": THETA_NZ,
            "carbon_limit_nz": cap_nz,
            "carbon_footprint_vw": cf_vw,
            "carbon_footprint_vw_nz": float(w_nz @ carbon),
            "slack": cap_nz - float(w_nz @ carbon),
            "feasible_within_1e-6": bool((cap_nz - float(w_nz @ carbon)) >= -1e-6),
            "optimizer_success": bool(nz_ok),
        })

        print(
            f"{year}: cap_NZ={cap_nz:7.2f} | "
            f"CF vw={cf_vw:7.2f}  vw_nz={float(w_nz @ carbon):7.2f}"
        )

    # ===================================================================== #
    # === Assemble Part 4 monthly panel                                   == #
    # ===================================================================== #
    monthly = pd.DataFrame(monthly_series).sort_index()
    monthly = monthly.loc[f"{PERF_START}-01-01":f"{PERF_END}-12-31"]
    monthly["vw"] = vw_monthly.reindex(monthly.index)
    monthly = monthly[["vw", "vw_drift", "vw_nz"]]

    annual = pd.DataFrame(annual_rows)
    weights = pd.DataFrame(weight_rows)
    te_table = pd.DataFrame(te_rows)
    slack_table = pd.DataFrame(slack_rows)
    path_table = pd.DataFrame(path_rows)

    rf_monthly = _load_rf(RF_PATH)

    # ===================================================================== #
    # === Read Part 3 VW(0.5) cache for joint comparison                 == #
    # ===================================================================== #
    cache_vw50 = OUTPUTS / "_cache_part3_vw50_monthly.csv"
    cache_vw50_annual = OUTPUTS / "_cache_part3_vw_vw50_annual.csv"
    if not cache_vw50.exists() or not cache_vw50_annual.exists():
        raise RuntimeError(
            "Part 4 requires the VW(0.5) outputs from Part 3. "
            "Run `python src/05_part3.py` first."
        )
    vw50_monthly = pd.read_csv(cache_vw50, parse_dates=["date"]).set_index("date")
    vw50_annual = pd.read_csv(cache_vw50_annual)

    # Joint monthly panel (vw, vw_50, vw_nz)
    joint_monthly = pd.DataFrame({
        "vw":    monthly["vw"],
        "vw_50": vw50_monthly["vw_50"].reindex(monthly.index),
        "vw_nz": monthly["vw_nz"],
    })

    stats = summary_stats(joint_monthly, rf=rf_monthly, benchmark="vw")

    # ===================================================================== #
    # === Persist all deliverables                                        == #
    # ===================================================================== #
    monthly[["vw", "vw_nz"]].to_csv(
        TABLES / "part4_returns_vw_netzero.csv", index_label="date"
    )
    stats.to_csv(TABLES / "part4_summary_vw_vs_carbon50_vs_netzero.csv", index=False)
    weights[weights["portfolio"] == "vw_nz"].to_csv(
        TABLES / "part4_weights_vw_netzero.csv", index=False
    )
    path_table.to_csv(
        TABLES / "part4_netzero_target_vs_realized_cf.csv", index=False
    )
    te_table[te_table["portfolio"] == "vw_nz"].to_csv(
        TABLES / "part4_tracking_error_vw_netzero.csv", index=False
    )
    slack_table[slack_table["portfolio"] == "vw_nz"].to_csv(
        TABLES / "part4_constraint_slack_vw_netzero.csv", index=False
    )

    # --- feasibility / optimizer status report --------------------------- #
    worst = slack_table[slack_table["portfolio"] == "vw_nz"]["slack"].min()
    if worst < -1e-6:
        print(f"WARNING: Part 4 NZ cap violated in at least one year: min slack = {worst:.6f}")
    else:
        print(f"Part 4 NZ-cap check OK: min slack = {worst:.6f} tCO2e/M$")
    failed = slack_table[~slack_table["optimization_success"]]
    if len(failed):
        print("WARNING: Part 4 optimizer failed for:")
        print(failed[["year", "portfolio", "optimization_message"]].to_string(index=False))

    # ===================================================================== #
    # === Figures                                                         == #
    # ===================================================================== #
    plot_cumulative(
        joint_monthly, ["vw", "vw_50", "vw_nz"],
        FIGURES / "part4_cumulative_vw_vs_carbon50_vs_netzero.png",
        "Section 4 - Cumulative growth of $1: VW vs VW(-50%) vs VW(NZ)",
    )

    # --- CF target vs realised ----------------------------------------- #
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(path_table["year"], path_table["carbon_limit_nz"],
            marker="o", linewidth=2, label="NZ cap target")
    ax.plot(path_table["year"], path_table["carbon_footprint_vw_nz"],
            marker="s", linewidth=2, label="Realised CF (VW(NZ))")
    ax.plot(path_table["year"], path_table["carbon_footprint_vw"],
            marker="^", linewidth=2, linestyle="--", label="CF(VW) (benchmark)")
    ax.set_title("Section 4.1 - Net-Zero cap path vs realised carbon footprint")
    ax.set_xlabel("Allocation year")
    ax.set_ylabel("CF (tCO2e per USD-million invested)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "part4_cf_target_vs_realized.png", dpi=160)
    plt.close(fig)

    # --- joint WACI plot ----------------------------------------------- #
    # Re-attach a `vw_50` series from the Part 3 cache.
    annual_view = annual.copy()
    annual_view["portfolio"] = annual_view["portfolio"].replace({"vw_drift": "vw"})
    vw50_for_plot = vw50_annual[vw50_annual["portfolio"] == "vw_50"][[
        "year", "portfolio",
        "waci_tco2e_per_musd_revenue",
        "carbon_footprint_tco2e_per_musd_invested",
    ]]
    waci_panel = pd.concat([
        annual_view[["year", "portfolio",
                     "waci_tco2e_per_musd_revenue",
                     "carbon_footprint_tco2e_per_musd_invested"]],
        vw50_for_plot,
    ], ignore_index=True)

    def _plot_panel(metric: str, path, title: str, ylabel: str) -> None:
        pivot = waci_panel.pivot(index="year", columns="portfolio", values=metric)
        cols = [c for c in ["vw", "vw_50", "vw_nz"] if c in pivot.columns]
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot[cols].plot(ax=ax, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Allocation year")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    _plot_panel(
        "waci_tco2e_per_musd_revenue",
        FIGURES / "part4_waci_vw_vs_carbon50_vs_netzero.png",
        "Section 4 - WACI: VW vs VW(-50%) vs VW(NZ)",
        "WACI (tCO2e per USD-million revenue)",
    )

    # --- TE of VW(NZ) -------------------------------------------------- #
    te_nz = te_table[te_table["portfolio"] == "vw_nz"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(te_nz["year"], te_nz["ex_ante_tracking_error_annual"],
            marker="o", linewidth=2, label="ex-ante")
    ax.plot(te_nz["year"], te_nz["ex_post_tracking_error_annual"],
            marker="s", linewidth=2, linestyle="--", label="ex-post")
    ax.set_title("Section 4 - Annualised tracking error of VW(NZ) vs VW")
    ax.set_xlabel("Allocation year")
    ax.set_ylabel("Tracking error (annualised)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "part4_tracking_error_vw_netzero.png", dpi=160)
    plt.close(fig)

    print("\nPart 4 performance summary:")
    print(stats.to_string(index=False))
    print(f"\nSaved Part IV deliverables to {OUTPUTS / 'tables'} and {FIGURES}")


if __name__ == "__main__":
    run()

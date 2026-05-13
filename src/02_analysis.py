import pandas as pd
import numpy as np
import os


def run_financial_analysis():

    # ── PART 1: Carbon Intensity at end-2013 ─────────────────────────────────
    try:
        rev = pd.read_csv('data/processed/Clean_Revenues_Pacific.csv')
        co2 = pd.read_csv('data/processed/Clean_CO2_Scope1_Pacific.csv')

        merged = pd.merge(co2[['ISIN', '2013']], rev[['ISIN', '2013']], on='ISIN')
        merged = merged[merged['2013_y'] > 0].copy()
        merged['CI_2013'] = merged['2013_x'] / (merged['2013_y'] / 1000)
        merged[['ISIN', 'CI_2013']].to_csv('data/processed/final_ci_scores.csv', index=False)
        print(f"CI scores calculated for {len(merged)} assets.")
    except Exception as e:
        print(f"Error in CI calculation: {e}")

    # ── PART 2: Price cleaning + returns + universe (estimation 1999–2013) ────
    try:
        # --- Load and reshape ---
        prices_raw = pd.read_csv('data/processed/Clean_Prices_Pacific.csv')
        prices = (
            prices_raw
            .set_index('ISIN')
            .drop(columns=['NAME'], errors='ignore')
            .T
        )
        prices.index = pd.to_datetime(prices.index)
        prices = prices.apply(pd.to_numeric, errors='coerce').sort_index()

        # --- Rule: prices < 0.5 are invalid (create extreme returns) ---
        prices[prices < 0.5] = np.nan

        # --- Rule: missing values ---
        # Trailing NaN (after last valid price) = delisting → set to 0
        # so the next computed return is exactly −100%.
        # Middle NaN = data gap → forward-fill with last valid price.
        # Leading NaN = not yet listed → leave as NaN (excluded from estimation).
        non_nan        = prices.notna().astype(np.int8)
        cum_from_below = non_nan.iloc[::-1].cumsum(axis=0).iloc[::-1]
        trailing_mask  = prices.isna() & (cum_from_below == 0)
        prices         = prices.ffill(axis=0)       # middle NaN → ffill
        prices[trailing_mask] = 0.0                  # trailing NaN → delisting

        # --- Rule: simple returns, safe division ---
        prev   = prices.shift(1)
        ret    = (prices / prev.replace(0.0, np.nan)) - 1.0
        ret[(prices == 0.0) & (prev > 0.0)] = -1.0  # delisting = −100%
        ret    = ret.replace([np.inf, -np.inf], np.nan)

        # --- Rule: investment universe at end-2013 for implementation in 2014 ---
        cutoff_date = '2013-12-31'
        # Use full history up to end-2013 for all filter checks.
        ret_window = ret.loc[:cutoff_date]

        # History filter: >= 36 non-NaN monthly returns
        obs_count = ret_window.count()

        # Stale filter: >50% zero returns over any rolling 120-month window.
        # Denominator = actual observations, NOT total rows (avoids bias for
        # short-listed firms that have many leading NaN).
        is_zero        = (ret_window == 0.0).astype(np.float32)
        roll_zeros     = is_zero.rolling(window=120, min_periods=1).sum()
        roll_obs       = (
            ret_window.notna().astype(np.float32)
                      .rolling(window=120, min_periods=1).sum()
        )
        stale_ratio    = roll_zeros / roll_obs.replace(0.0, np.nan)
        stale_flags    = (stale_ratio > 0.50).any()   # ever stale in any window

        # Carbon filter: firm must have Scope-1 OR Scope-2 data at end-2013
        has_co2 = pd.Series(False, index=ret_window.columns)
        for co2_path in [
            'data/processed/Clean_CO2_Scope1_Pacific.csv',
            'data/processed/Clean_CO2_Scope2_Pacific.csv',
        ]:
            if os.path.exists(co2_path):
                co2_df = pd.read_csv(co2_path).set_index('ISIN')
                co2_df.columns = co2_df.columns.astype(str)
                if '2013' in co2_df.columns:
                    coverage = co2_df['2013'].reindex(ret_window.columns).notna()
                    has_co2  = has_co2 | coverage

        # Combine all three filters and persist a reproducible snapshot table.
        eligible_history = obs_count >= 36
        eligible_not_stale = ~stale_flags
        eligible_carbon = has_co2
        included_final = eligible_history & eligible_not_stale & eligible_carbon

        snapshot = pd.DataFrame({
            'ISIN': ret_window.columns,
            'eligible_history_36m': eligible_history.reindex(ret_window.columns).astype(bool).values,
            'eligible_not_stale_120m': eligible_not_stale.reindex(ret_window.columns).astype(bool).values,
            'eligible_carbon_scope1_or_scope2': eligible_carbon.reindex(ret_window.columns).astype(bool).values,
            'included_final': included_final.reindex(ret_window.columns).astype(bool).values,
            'cutoff_date': cutoff_date,
        })
        snapshot_path = 'data/processed/part1_universe_snapshot.csv'
        snapshot.to_csv(snapshot_path, index=False)

        summary = pd.DataFrame([
            {'rule': 'history_lt_36m', 'removed_count': int((~eligible_history).sum())},
            {'rule': 'stale_prices_over_50pct_in_any_120m_window', 'removed_count': int((~eligible_not_stale).sum())},
            {'rule': 'missing_scope1_and_scope2_at_2013', 'removed_count': int((~eligible_carbon).sum())},
            {'rule': 'final_included', 'removed_count': int(included_final.sum())},
            {'rule': 'final_excluded', 'removed_count': int((~included_final).sum())},
        ])
        summary_path = 'outputs/part1_universe_filter_summary.csv'
        summary.to_csv(summary_path, index=False)

        print(f"PART1_UNIVERSE_FIXED | cutoff={cutoff_date} | snapshot={snapshot_path} | summary={summary_path}")

        # Downstream optimization input: always reload the ISIN list from snapshot.
        included_isins = (
            pd.read_csv(snapshot_path)
            .loc[lambda df: df['included_final'], 'ISIN']
            .astype(str)
            .tolist()
        )

        # Save: return series for valid firms, full history up to end-2013
        ret_window[included_isins].to_csv('data/processed/final_returns_matrix.csv')
        print(f"Analysis complete: {len(included_isins)} assets in universe for 2014.")

        # Save exclusion log for traceability
        excluded = snapshot.loc[~snapshot['included_final'], 'ISIN']
        log_rows = []
        for isin in excluded:
            reasons = []
            if obs_count[isin] < 36:
                reasons.append(f"insufficient_history({int(obs_count[isin])}mo)")
            if stale_flags.get(isin, False):
                reasons.append("stale_prices")
            if not has_co2.get(isin, False):
                reasons.append("no_carbon_data")
            log_rows.append({'ISIN': isin, 'reason': ' | '.join(reasons)})
        pd.DataFrame(log_rows).to_csv('data/processed/excluded_firms_2013.csv', index=False)
        print(f"Exclusion log saved: data/processed/excluded_firms_2013.csv")

    except Exception as e:
        import traceback
        print(f"Error in returns analysis: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_financial_analysis()

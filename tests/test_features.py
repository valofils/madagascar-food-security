"""
tests/test_features.py
-----------------------
Unit tests for src/features.py.

Tests cover lag feature correctness, rolling statistics, unit-level
aggregations, categorical encoding, and the full feature matrix builder.
All tests use synthetic in-memory DataFrames — no files required.

Run:
    pytest tests/test_features.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import (
    add_lag_features,
    add_unit_features,
    encode_categoricals,
    build_feature_matrix,
    FEATURE_COLS,
    TARGET_COL,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_df(n_periods=6, fnid="MG001", phase_seq=None, scale="IPC3"):
    """
    Build a minimal synthetic DataFrame that mimics the preprocessed output.
    phase_seq: list of IPC phases (length == n_periods). Defaults to 1..n.
    """
    if phase_seq is None:
        phase_seq = [1, 1, 2, 3, 2, 1][:n_periods]

    dates = pd.date_range("2020-01-01", periods=n_periods, freq="2MS")
    return pd.DataFrame({
        "fnid":             [fnid] * n_periods,
        "projection_start": dates,
        "ipc_phase":        phase_seq,
        "scale":            [scale] * n_periods,
        "year":             dates.year,
        "month":            dates.month,
        "quarter":          dates.quarter,
        "is_lean_season":   [1, 0, 1, 0, 1, 0][:n_periods],
        "period_days":      [60] * n_periods,
        "preference_rating": [90.0] * n_periods,
    })


def make_multi_unit_df():
    """Two geographic units with different risk profiles."""
    df1 = make_df(fnid="MG001", phase_seq=[1, 1, 2, 2, 3, 3])
    df2 = make_df(fnid="MG002", phase_seq=[3, 3, 3, 4, 4, 3])
    return pd.concat([df1, df2], ignore_index=True)


# ── FEATURE_COLS completeness ─────────────────────────────────────────────────

class TestFeatureCols:
    def test_feature_cols_has_16_features(self):
        assert len(FEATURE_COLS) == 16

    def test_feature_cols_contains_expected_keys(self):
        expected = [
            "year", "month", "quarter", "is_lean_season", "period_days",
            "lag_1", "lag_2", "rolling_mean_3", "rolling_max_3", "phase_trend",
            "unit_mean_phase", "unit_max_phase", "unit_pct_crisis",
            "unit_code", "is_ipc2", "preference_rating",
        ]
        assert FEATURE_COLS == expected

    def test_target_col_is_ipc_phase(self):
        assert TARGET_COL == "ipc_phase"


# ── add_lag_features ──────────────────────────────────────────────────────────

class TestLagFeatures:
    def test_lag_1_shifts_by_one_period(self):
        df = make_df(phase_seq=[1, 2, 3, 4, 2, 1])
        result = add_lag_features(df)
        # lag_1 for row index 1 should be phase from row 0
        assert result["lag_1"].iloc[1] == 1.0
        assert result["lag_1"].iloc[2] == 2.0
        assert result["lag_1"].iloc[3] == 3.0

    def test_lag_2_shifts_by_two_periods(self):
        df = make_df(phase_seq=[1, 2, 3, 4, 2, 1])
        result = add_lag_features(df)
        assert result["lag_2"].iloc[2] == 1.0
        assert result["lag_2"].iloc[3] == 2.0

    def test_first_row_lag_is_null(self):
        df = make_df(phase_seq=[1, 2, 3, 4, 2, 1])
        result = add_lag_features(df)
        assert pd.isna(result["lag_1"].iloc[0])
        assert pd.isna(result["lag_2"].iloc[0])
        assert pd.isna(result["lag_2"].iloc[1])

    def test_phase_trend_is_lag1_minus_lag2(self):
        df = make_df(phase_seq=[1, 2, 3, 4, 2, 1])
        result = add_lag_features(df)
        # at index 3: lag_1=3, lag_2=2, trend=1
        assert result["phase_trend"].iloc[3] == pytest.approx(1.0)
        # at index 4: lag_1=4, lag_2=3, trend=1
        assert result["phase_trend"].iloc[4] == pytest.approx(1.0)

    def test_rolling_mean_3_uses_previous_periods(self):
        df = make_df(phase_seq=[2, 4, 2, 4, 2, 4])
        result = add_lag_features(df)
        # rolling_mean_3 at index 3 uses lags of [2,4,2] -> mean=2.67
        assert result["rolling_mean_3"].iloc[3] == pytest.approx(8/3, abs=0.01)

    def test_rolling_max_3_uses_previous_periods(self):
        df = make_df(phase_seq=[1, 3, 2, 4, 1, 2])
        result = add_lag_features(df)
        # rolling_max_3 at index 3: previous 3 = [3,2,4]... shifted, so [1,3,2] -> max=3
        assert result["rolling_max_3"].iloc[3] == pytest.approx(3.0)

    def test_lags_are_per_unit_not_across_units(self):
        """lag_1 for MG002's first row must be NaN, not MG001's last phase."""
        df = make_multi_unit_df()
        result = add_lag_features(df)
        mg002_rows = result[result["fnid"] == "MG002"].reset_index(drop=True)
        assert pd.isna(mg002_rows["lag_1"].iloc[0])


# ── add_unit_features ─────────────────────────────────────────────────────────

class TestUnitFeatures:
    def test_unit_mean_phase_is_correct(self):
        df = make_df(fnid="MG001", phase_seq=[1, 2, 3, 2, 1, 3])
        result = add_unit_features(df)
        expected_mean = np.mean([1, 2, 3, 2, 1, 3])
        assert result["unit_mean_phase"].iloc[0] == pytest.approx(expected_mean)

    def test_unit_max_phase_is_correct(self):
        df = make_df(fnid="MG001", phase_seq=[1, 2, 3, 2, 1, 3])
        result = add_unit_features(df)
        assert result["unit_max_phase"].iloc[0] == 3.0

    def test_unit_pct_crisis_is_correct(self):
        # 2 out of 6 periods are Phase 3+
        df = make_df(fnid="MG001", phase_seq=[1, 2, 3, 2, 1, 3])
        result = add_unit_features(df)
        assert result["unit_pct_crisis"].iloc[0] == pytest.approx(2/6)

    def test_high_risk_unit_has_higher_pct_crisis(self):
        df = make_multi_unit_df()
        result = add_unit_features(df)
        mg001 = result[result["fnid"] == "MG001"]["unit_pct_crisis"].iloc[0]
        mg002 = result[result["fnid"] == "MG002"]["unit_pct_crisis"].iloc[0]
        assert mg002 > mg001

    def test_unit_stats_consistent_across_all_rows_of_unit(self):
        """Every row for a given fnid must have the same unit stats."""
        df = make_df(fnid="MG001", phase_seq=[1, 2, 3, 2, 1, 3])
        result = add_unit_features(df)
        assert result["unit_mean_phase"].nunique() == 1
        assert result["unit_max_phase"].nunique() == 1
        assert result["unit_pct_crisis"].nunique() == 1


# ── encode_categoricals ───────────────────────────────────────────────────────

class TestEncodeCategoricals:
    def test_unit_code_is_integer(self):
        df = make_df()
        result = encode_categoricals(df)
        assert pd.api.types.is_integer_dtype(result["unit_code"])

    def test_is_ipc2_flag_set_for_ipc2_scale(self):
        df = make_df(scale="IPC2")
        result = encode_categoricals(df)
        assert (result["is_ipc2"] == 1).all()

    def test_is_ipc2_flag_unset_for_ipc3_scale(self):
        df = make_df(scale="IPC3")
        result = encode_categoricals(df)
        assert (result["is_ipc2"] == 0).all()

    def test_different_units_get_different_codes(self):
        df = make_multi_unit_df()
        result = encode_categoricals(df)
        codes = result.groupby("fnid")["unit_code"].first()
        assert codes["MG001"] != codes["MG002"]


# ── build_feature_matrix ──────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_output_contains_all_feature_cols(self):
        df = make_multi_unit_df()
        result = build_feature_matrix(df, drop_nulls=False)
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_output_contains_target_col(self):
        df = make_multi_unit_df()
        result = build_feature_matrix(df, drop_nulls=False)
        assert TARGET_COL in result.columns

    def test_drop_nulls_removes_rows_with_null_features(self):
        df = make_multi_unit_df()
        full  = build_feature_matrix(df, drop_nulls=False)
        clean = build_feature_matrix(df, drop_nulls=True)
        # dropping nulls must produce fewer or equal rows
        assert len(clean) <= len(full)

    def test_no_nulls_in_features_after_drop(self):
        df = make_multi_unit_df()
        result = build_feature_matrix(df, drop_nulls=True)
        null_counts = result[FEATURE_COLS].isnull().sum()
        assert null_counts.sum() == 0, \
            f"Null values found in features:\n{null_counts[null_counts > 0]}"

    def test_phase_values_are_in_valid_range(self):
        df = make_multi_unit_df()
        result = build_feature_matrix(df, drop_nulls=True)
        assert result[TARGET_COL].between(1, 5).all()

    def test_is_lean_season_is_binary(self):
        df = make_multi_unit_df()
        result = build_feature_matrix(df, drop_nulls=True)
        assert set(result["is_lean_season"].unique()).issubset({0, 1})

    def test_row_count_is_positive(self):
        df = make_multi_unit_df()
        result = build_feature_matrix(df, drop_nulls=True)
        assert len(result) > 0
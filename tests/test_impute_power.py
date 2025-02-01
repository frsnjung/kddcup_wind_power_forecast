import numpy as np
import pandas as pd
import pytest

from windfarm_forecast.feature_engineering import impute_power_output


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    df = pd.DataFrame(
        {
            "timestamp": ["2021-01-01 00:00"] * 4 + ["2021-01-01 00:10"] * 4,
            "TurbID": [1, 2, 3, 4] * 2,
            "Patv": [100, 200, np.nan, 400, 150, np.nan, 300, 450],
            "impute_day_patv": [0, 0, 1, 0, 0, 1, 0, 0],
            "other_col": ["a"] * 8,
        }
    )
    return df


@pytest.fixture
def similar_turbines_data():
    """Fixture providing similar turbines data for testing."""
    df = pd.DataFrame(
        {
            "turbine_id": [3, 3, 3, 2, 2],
            "rank": [1, 2, 3, 1, 2],
            "similar_turbine_id": [1, 4, 2, 1, 3],
            "correlation": [0.9, 0.8, 0.7, 0.9, 0.8],
        }
    )
    return df


def test_original_columns_preserved(sample_data, similar_turbines_data):
    """Test that original columns are preserved in the output."""
    result_df = impute_power_output(sample_data, similar_turbines_data)
    assert set(sample_data.columns).issubset(set(result_df.columns))


def test_non_imputed_values_unchanged(sample_data, similar_turbines_data):
    """Test that non-imputed values remain unchanged."""
    result_df = impute_power_output(sample_data, similar_turbines_data)
    mask = sample_data["impute_day_patv"] == 0
    pd.testing.assert_series_equal(result_df.loc[mask, "Patv"], sample_data.loc[mask, "Patv"], check_names=False)


def test_all_required_values_imputed(sample_data, similar_turbines_data):
    """Test that all values that needed imputation were imputed."""
    result_df = impute_power_output(sample_data, similar_turbines_data)
    mask = sample_data["impute_day_patv"] == 1
    assert result_df.loc[mask, "Patv_imputed"].notna().all()


def test_imputed_values_correct(sample_data, similar_turbines_data):
    """Test that imputed values are correctly calculated as average of similar turbines."""
    result_df = impute_power_output(sample_data, similar_turbines_data)

    # Test first imputed value (Turbine 3, first timestamp)
    mask = (sample_data["TurbID"] == 3) & (sample_data["timestamp"] == "2021-01-01 00:00")
    similar_mask = (sample_data["TurbID"].isin([1, 4, 2])) & (sample_data["timestamp"] == "2021-01-01 00:00")

    expected_value = sample_data.loc[similar_mask, "Patv"].mean()
    actual_value = result_df.loc[mask, "Patv_imputed"].iloc[0]

    assert np.isclose(actual_value, expected_value)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # Test case 1: Empty DataFrame, expecting result.empty to be True
        (pd.DataFrame(columns=["timestamp", "TurbID", "Patv", "impute_day_patv"]), True),
        # Test case 2: DataFrame with no imputation needed, expecting result.empty to be False
        (
            pd.DataFrame(
                {
                    "timestamp": ["2021-01-01"] * 3,
                    "TurbID": [1, 2, 3],
                    "Patv": [100, 200, 300],
                    "impute_day_patv": [0, 0, 0],
                }
            ),
            False,
        ),
    ],
)
def test_edge_cases(test_input, expected, similar_turbines_data):
    """Test edge cases of the imputation function."""
    result = impute_power_output(test_input, similar_turbines_data)
    assert result.empty == expected


def test_missing_similar_turbines(sample_data):
    """Test behavior when similar turbines data is missing."""
    empty_similar = pd.DataFrame(
        {
            "turbine_id": [],
            "rank": pd.Series([], dtype="int64"),  # Specify numeric dtype
            "similar_turbine_id": [],
            "correlation": [],
        }
    )
    result_df = impute_power_output(sample_data, empty_similar)
    # Should still have Patv_imputed column but with NaN values where imputation was needed
    mask = sample_data["impute_day_patv"] == 1
    assert result_df.loc[mask, "Patv_imputed"].isna().all()


def test_all_similar_turbines_need_imputation(similar_turbines_data):
    """Test case where all similar turbines also need imputation."""
    df = pd.DataFrame(
        {
            "timestamp": ["2021-01-01"] * 4,
            "TurbID": [1, 2, 3, 4],  # Include all turbine IDs referenced in similar_turbines_data
            "Patv": [np.nan, np.nan, np.nan, np.nan],
            "impute_day_patv": [1, 1, 1, 1],
        }
    )

    result_df = impute_power_output(df, similar_turbines_data)
    assert result_df["Patv_imputed"].isna().all()

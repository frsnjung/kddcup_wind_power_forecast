import pandas as pd


def impute_power_output(df: pd.DataFrame, similar_turbines_df: pd.DataFrame, n_similar: int = 10) -> pd.DataFrame:
    """
    Iimpute the power output using the average of similar turbines.

    Args:
        df: DataFrame containing power output data
        similar_turbines_df: DataFrame with similar turbine information
        n_similar: Number of similar turbines to use for imputation

    Returns:
        DataFrame with imputed power values
    """
    df = df.copy()

    # Early exit if no imputation needed
    if not (df["impute_day_patv"] == 1).any():
        print("No values to impute")
        return df

    # Prepare similar turbines lookup
    similar_lookup = (
        similar_turbines_df.nsmallest(n_similar, "rank", keep="all")
        .groupby("turbine_id")["similar_turbine_id"]
        .agg(list)
        .to_dict()
    )

    # Create a pivot table of power values
    power_pivot = df.pivot(index="timestamp", columns="TurbID", values="Patv")

    # Create a pivot table of imputation flags
    impute_pivot = df.pivot(index="timestamp", columns="TurbID", values="impute_day_patv")

    # Get locations needing imputation
    impute_mask = impute_pivot == 1

    # Iterate through turbines that need imputation (should be much fewer iterations)
    turbines_to_impute = impute_mask.columns[impute_mask.any()]

    for turbine_id in turbines_to_impute:
        # Get similar turbines for this turbine
        similar_turbines = similar_lookup.get(turbine_id, [])
        if not similar_turbines:
            continue

        # Get power values of similar turbines
        similar_powers = power_pivot[similar_turbines]

        # Calculate mean power of similar turbines (excluding NaN)
        imputed_values = similar_powers.mean(axis=1)

        # Get timestamps where this turbine needs imputation
        timestamps_to_impute = impute_mask.index[impute_mask[turbine_id]]

        # Update the power pivot table with imputed values
        power_pivot.loc[timestamps_to_impute, turbine_id] = imputed_values.loc[timestamps_to_impute]

    # Convert pivot table back to original format
    df_imputed = power_pivot.reset_index().melt(id_vars=["timestamp"], var_name="TurbID", value_name="Patv_imputed")

    # Merge back with original dataframe to keep all columns including original Patv
    df_imputed = df.merge(df_imputed, on=["timestamp", "TurbID"], how="left")

    return df_imputed

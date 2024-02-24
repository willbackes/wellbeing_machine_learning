import pandas as pd


def merge_data(household_data, individual_data):
    merge_household_df = _merge_household_data(household_data)
    merge_individual_df = _merge_individual_data(individual_data)
    return _merge_individual_and_household(merge_household_df, merge_individual_df)


def _merge_household_data(household_data):
    merge_household_df = household_data[0]
    for df in household_data[1:]:
        merge_household_df = merge_household_df.merge(
            df,
            on=["cid", "syear"],
            how="outer",
        )
    return merge_household_df


def _merge_individual_data(individual_data):
    merge_individual_df = individual_data[0]
    for df in individual_data[1:]:
        merge_individual_df = merge_individual_df.merge(
            df,
            on=["pid", "cid", "syear"],
            how="outer",
        )
    return merge_individual_df


def _merge_individual_and_household(merge_household_df, merge_individual_df):
    return pd.merge(
        merge_household_df,
        merge_individual_df,
        on=["cid", "syear"],
        how="right",
    )

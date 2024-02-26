import pandas as pd


def merge_household_data(household_data):
    merge_household_df = household_data[0]
    for df in household_data[1:]:
        merge_household_df = merge_household_df.merge(
            df,
            on=["hid", "syear"],
            how="outer",
        )
    return merge_household_df


def merge_individual_data(individual_data):
    merge_individual_df = individual_data[0]
    for df in individual_data[1:]:
        merge_individual_df = merge_individual_df.merge(
            df,
            on=["pid", "hid", "syear"],
            how="outer",
        )
    return merge_individual_df


def merge_individual_and_household(household_df, individual_df):
    return pd.merge(
        household_df,
        individual_df,
        on=["hid", "syear"],
        how="right",
    )

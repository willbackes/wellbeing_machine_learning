import pandas as pd


def merge_household_or_individual_datasets(data, on):
    merged_df = data[0]
    for df in data[1:]:
        merged_df = merged_df.merge(
            df,
            on=on,
            how="outer",
        )
    return merged_df


def merge_individual_and_household(household_df, individual_df):
    return pd.merge(
        household_df,
        individual_df,
        on=["hid", "syear"],
        how="right",
    )

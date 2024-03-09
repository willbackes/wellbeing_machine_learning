"""" Functions to merge the household and individual datasets."""

import pandas as pd


def merge_household_or_individual_datasets(data, on):
    """Merges a list of pandas DataFrames on a given column or columns.

    This function performs an outer join on the DataFrames in the list, using the column(s) specified by `on`.
    It checks the input parameters for validity before performing the merge.

    Args:
        data (list of pd.DataFrame): A list of pandas DataFrames to merge.
        on (str or list of str): Column or list of columns to merge on.

    Returns:
        pd.DataFrame: The merged DataFrame.

    Raises:
        ValueError: If `data` is not a list of DataFrames, if `data` contains less than two DataFrames,
                    if `on` is not a string or a list of strings, or if any DataFrame in `data` does not contain `on`.

    """
    _fail_if_not_a_list_of_dataframes(data)
    _fail_if_less_than_two_dataframes(data)
    _fail_if_not_a_dataframe(data)
    _fail_if_not_string_or_list_of_strings(on)
    merged_df = data[0]
    for df in data[1:]:
        merged_df = merged_df.merge(
            df,
            on=on,
            how="outer",
        )
    return merged_df


def merge_individual_and_household(household_df, individual_df):
    """Merges household and individual DataFrames on 'hid' and 'syear' columns.

    This function performs a right join on the DataFrames, using the 'hid' and 'syear' columns.
    It checks the input parameters for validity before performing the merge.

    Args:
        household_df (pd.DataFrame): The household DataFrame.
        individual_df (pd.DataFrame): The individual DataFrame.

    Returns:
        pd.DataFrame: The merged DataFrame.

    Raises:
        ValueError: If `household_df` or `individual_df` is not a DataFrame, or if 'hid' or 'syear' columns are missing.

    """
    _fail_if_not_pandas_dataframe(household_df, individual_df)
    _fail_if_hid_or_syear_columns_missing(household_df, individual_df)
    return pd.merge(
        household_df,
        individual_df,
        on=["hid", "syear"],
        how="right",
    )


def _fail_if_not_a_list_of_dataframes(data):
    if not isinstance(data, list):
        msg = "Data must be a list of pandas DataFrames."
        raise TypeError(msg)


def _fail_if_less_than_two_dataframes(data):
    if len(data) < 2:
        msg = "Data must be a list of at least two pandas DataFrames."
        raise TypeError(msg)


def _fail_if_not_a_dataframe(data):
    if not all(isinstance(df, pd.DataFrame) for df in data):
        msg = "All elements in data must be pandas DataFrames."
        raise TypeError(msg)


def _fail_if_not_string_or_list_of_strings(on):
    if not isinstance(on, str | list) or (
        isinstance(on, list) and not all(isinstance(item, str) for item in on)
    ):
        msg = "on must be a string or a list of strings."
        raise TypeError(msg)


def _fail_if_not_pandas_dataframe(household_df, individual_df):
    if not isinstance(household_df, pd.DataFrame) or not isinstance(
        individual_df,
        pd.DataFrame,
    ):
        msg = "Both household_df and individual_df must be pandas DataFrames."
        raise TypeError(
            msg,
        )


def _fail_if_hid_or_syear_columns_missing(household_df, individual_df):
    if not {"hid", "syear"}.issubset(household_df.columns) or not {
        "hid",
        "syear",
    }.issubset(individual_df.columns):
        msg = "Both household_df and individual_df must contain 'hid' and 'syear' columns."
        raise ValueError(
            msg,
        )

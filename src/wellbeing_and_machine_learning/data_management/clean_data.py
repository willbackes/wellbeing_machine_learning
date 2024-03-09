"""" This module contains functions to clean and transform the input data."""

import numpy as np
import pandas as pd
from pandas import NA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def clean_data(merged_data):
    """Cleans and transforms the input DataFrame.

    This function filters the data by year, cleans invalid data, and creates new columns by transforming and
    extracting data from existing columns. It also converts the data types of several columns.

    Args:
        merged_data (pd.DataFrame): The input DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.

    """
    _fail_if_not_a_dataframe(merged_data)
    valid_data = pd.DataFrame()
    valid_data = merged_data[
        (merged_data["syear"] >= 2010) & (merged_data["syear"] <= 2018)
    ]
    valid_data = _clean_invalid_data(valid_data)
    valid_data["hghinc"] = _positive_number_only(valid_data["hghinc"]).astype(
        pd.Float64Dtype(),
    )

    df = pd.DataFrame()
    df["pid"] = valid_data["pid"]
    df["hid"] = valid_data["hid"]
    df["syear"] = valid_data["syear"].astype(pd.UInt16Dtype())

    df["birthyear"] = _positive_number_only(valid_data["gebjahr"]).astype(
        pd.UInt16Dtype(),
    )
    df["age"] = np.subtract(df["syear"], df["birthyear"])
    df["agesquared"] = df["age"] ** 2
    df["bmi"] = _positive_number_only(valid_data["bmi"])
    df["education"] = valid_data["pgbilzeit"].astype(pd.Float32Dtype())
    df["logincome"] = np.log(valid_data["hghinc"])
    df["health"] = valid_data["m11127"].astype(pd.UInt16Dtype())
    df["workinghours"] = valid_data["plb0183"].astype(pd.Float64Dtype())
    df["numberofpeople"] = (
        valid_data["hhgr"]
        .replace("[0] Aufgeloeste/n.bearbeitete Haushalte", NA)
        .astype(pd.UInt8Dtype())
    )
    df["numberofchildren"] = valid_data["k_nrkid"].astype(pd.UInt8Dtype())

    df["smonth"] = valid_data["pmonin"]
    df["birthregion"] = valid_data["birthregion"]
    df["migback"] = valid_data["migback"]
    df["housingstatus"] = valid_data["hlf0001_v3"]
    df["maritalstatus"] = _clean_marital_status(valid_data["pgfamstd"])
    df["religion"] = valid_data["plh0258_h"]

    df["disability"] = _extract_number_from_brackets(valid_data["m11124"])
    df["labourstatus"] = _clean_binary_data(
        valid_data["pglfs"],
        is_one="[11] Erwerbstätig",
    )
    df["sex"] = _extract_number_from_brackets(valid_data["sex"]) - 1

    df["lifesatisfaction"] = _extract_number_from_brackets(valid_data["plh0182"])

    return df


def convert_categorical_to_dummy(data, columns):
    """Converts categorical variables into dummy/indicator variables.

    This function uses one-hot encoding to convert categorical variables into a format that could be
    provided to machine learning algorithms to improve prediction performance.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (list of str): The names of the columns to convert.

    Returns:
        pd.DataFrame: A DataFrame with one-hot encoded variables.

    """
    _fail_if_not_a_dataframe(data)
    _fail_if_not_a_list_of_strings(columns)
    _fail_if_not_categorical_data(data, columns)
    encoder = OneHotEncoder(sparse_output=False)
    return pd.DataFrame(
        encoder.fit_transform(data[columns]),
        columns=encoder.get_feature_names_out(columns),
        index=data.index,
    )


def observed_means_for_missing_values(data, columns):
    """Fills missing values in specified columns with the mean value of each column.

    This function uses the SimpleImputer class from sklearn.impute module to fill missing values.
    The strategy used is "mean", which means that missing values are filled with the mean value of each column.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (list of str): The names of the columns to fill missing values in.

    Returns:
        pd.DataFrame: A DataFrame with missing values filled.

    """
    _fail_if_not_a_dataframe(data)
    _fail_if_not_a_list_of_strings(columns)
    imputer = SimpleImputer(strategy="mean")
    return pd.DataFrame(
        imputer.fit_transform(data[columns]),
        columns=columns,
        index=data.index,
    )


def _clean_invalid_data(data):
    """Replaces specific string values in the DataFrame with NA.

    This function replaces a set of predefined string values that represent invalid data with NA.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with invalid data replaced by NA.

    """
    _fail_if_not_a_dataframe(data)
    invalid_data_mapping = {
        "[-1] keine Angabe": NA,
        "[-2] trifft nicht zu": NA,
        "[-3] nicht valide": NA,
        "[-4] Unzulaessige Mehrfachantwort": NA,
        "[-5] in Fragebogenversion nicht enthalten": NA,
        "[-6] Fragebogenversion mit geaenderter Filterfuehrung": NA,
        "[-7] Nur in weniger eingeschränkter Edition verfügbar": NA,
        "[-8] Frage in diesem Jahr nicht Teil des Frageprogramms": NA,
    }
    return data.replace(invalid_data_mapping)


def _extract_number_from_brackets(data):
    """Extracts numbers enclosed in brackets from a pandas Series.

    This function uses a regular expression to extract numbers that are enclosed in brackets from each string in the Series.
    The extracted numbers are returned as a new Series with the same index as the input Series. The data type of the new Series is UInt16.

    Args:
        data (pd.Series): The input Series.

    Returns:
        pd.Series: A Series with the extracted numbers.

    """
    _fail_if_not_a_series(data)
    _fail_if_not_string_series(data)
    _fail_if_bracketed_numbers_missing(data)
    df = data.str.extract(r"\[(\d+)\]")
    return df.astype(pd.UInt16Dtype())


def _clean_binary_data(data, is_one):
    """Converts a pandas Series into binary format.

    This function applies a lambda function to each element in the Series. If an element is equal to the `is_one` parameter,
    it is replaced with 1; otherwise, it is replaced with 0. The result is a binary Series.

    Args:
        data (pd.Series): The input Series.
        is_one: The value in the Series to be replaced with 1.

    Returns:
        pd.Series: A binary Series.

    """
    _fail_if_not_a_series(data)
    _fail_if_not_string_series(data)
    _fail_if_is_one_not_in_data(data, is_one)
    df = data.apply(lambda x: 1 if x == is_one else 0)
    return df.astype(pd.UInt8Dtype())


def _clean_marital_status(data):
    """Cleans the marital status data in a pandas Series.

    This function replaces specific string values that represent invalid marital status data with NA.

    Args:
        data (pd.Series): The input Series containing marital status data.

    Returns:
        pd.Series: The Series with invalid marital status data replaced by NA.

    """
    _fail_if_not_a_series(data)
    _fail_if_not_string_series(data)
    invalid_marital_data = {
        "[6] Ehepartner im Ausland": NA,
        "[7] Eingetragene gleichgeschlechtliche Partnerschaft zusammenlebend": NA,
        "[8] Eingetragene gleichgeschlechtliche Partnerschaft getrennt lebend": NA,
    }
    return data.replace(invalid_marital_data)


def _positive_number_only(data):
    """Replaces non-positive values in a pandas Series with NA.

    This function uses the `where` method of the Series to replace all non-positive values (i.e., values less than or equal to 0) with NA.

    Args:
        data (pd.Series): The input Series.

    Returns:
        pd.Series: The Series with non-positive values replaced by NA.

    """
    _fail_if_not_a_series(data)
    return data.where(data > 0, NA)


def _fail_if_not_a_dataframe(data):
    if not isinstance(data, pd.DataFrame):
        msg = "Input data must be a pandas DataFrame."
        raise ValueError(msg)


def _fail_if_not_a_list_of_strings(columns):
    if not isinstance(columns, list) or not all(
        isinstance(col, str) for col in columns
    ):
        msg = "Input columns must be a list of strings."
        raise ValueError(msg)


def _fail_if_not_categorical_data(data, columns):
    if not all(isinstance(data[col].dtype, pd.CategoricalDtype) for col in columns):
        msg = "All specified columns must contain categorical data."
        raise ValueError(msg)


def _fail_if_not_a_series(data):
    if not isinstance(data, pd.Series):
        msg = "Input data must be a pandas Series."
        raise ValueError(msg)


def _fail_if_not_string_series(data):
    if not pd.api.types.is_string_dtype(data):
        msg = "Input Series must contain string data."
        raise ValueError(msg)


def _fail_if_bracketed_numbers_missing(data):
    if data.str.contains(r"\[\d+\]").sum() == 0:
        msg = "No bracketed numbers found in the input Series."
        raise ValueError(msg)


def _fail_if_is_one_not_in_data(data, is_one):
    if not data.empty and is_one not in data.values:
        msg = f"The value '{is_one}' is not found in the input Series."
        raise ValueError(msg)

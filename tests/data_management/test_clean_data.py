import numpy as np
import pandas as pd
from pandas import NA
from wellbeing_and_machine_learning.data_management.clean_data import (
    _clean_binary_data,
    _clean_invalid_data,
    _extract_number_from_brackets,
    _positive_number_only,
    convert_categorical_to_dummy,
    observed_means_for_missing_values,
)


def test_convert_categorical_to_dummy_columns():
    expected = {"color_Blue", "color_Green", "color_Red", "size_L", "size_M", "size_S"}
    data = pd.DataFrame(
        {
            "color": ["Red", "Blue", "Green", "Red", "Blue"],
            "size": ["S", "M", "L", "S", "M"],
        },
    )
    actual = convert_categorical_to_dummy(data, ["color", "size"])
    assert set(actual.columns) == expected


def test_convert_categorical_to_dummy_typical_result():
    expected = pd.DataFrame(
        {
            "color_Red": [1, 0, 0, 1, 0],
            "color_Green": [0, 0, 1, 0, 0],
            "color_Blue": [0, 1, 0, 0, 1],
            "size_L": [0, 0, 1, 0, 0],
            "size_M": [0, 1, 0, 0, 1],
            "size_S": [1, 0, 0, 1, 0],
        },
        dtype=pd.UInt8Dtype(),
    )
    data = pd.DataFrame(
        {
            "color": ["Red", "Blue", "Green", "Red", "Blue"],
            "size": ["S", "M", "L", "S", "M"],
        },
    )
    actual = convert_categorical_to_dummy(data, ["color", "size"]).astype(
        pd.UInt8Dtype(),
    )
    actual = actual.sort_index(axis=1)
    expected = expected.sort_index(axis=1)
    assert actual.equals(expected)


def test_observed_means_for_missing_values_check_missing_values():
    data = pd.DataFrame(
        {
            "age": [25, 30, 35, np.nan, 45],
            "income": [50000, 60000, np.nan, np.nan, 80000],
        },
    )
    actual = observed_means_for_missing_values(data, ["age", "income"])
    assert not actual.isnull().any().any()


def test_observed_means_for_missing_values_typical_result():
    expected = pd.DataFrame(
        {"age": [25, 30, 35, 33.75, 45], "income": [50000, 60000, 60000, 60000, 70000]},
    )
    data = pd.DataFrame(
        {
            "age": [25, 30, 35, np.nan, 45],
            "income": [50000, 60000, np.nan, np.nan, 70000],
        },
    )
    actual = observed_means_for_missing_values(data, ["age", "income"])
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_clean_invalid_data_number_of_changes():
    expected = 4
    data = pd.DataFrame(
        {
            "column1": ["[-1] keine Angabe", "valid data", "[-3] nicht valide"],
            "column2": [
                "[-2] trifft nicht zu",
                "[-4] Unzulaessige Mehrfachantwort",
                "valid data",
            ],
        },
    )
    actual = _clean_invalid_data(data)
    assert actual.isnull().sum().sum() == expected


def test_clean_invalid_data_typical_result():
    expected = pd.DataFrame(
        {"column1": [NA, "valid data", NA], "column2": [NA, NA, "valid data"]},
    )
    data = pd.DataFrame(
        {
            "column1": ["[-1] keine Angabe", "valid data", "[-3] nicht valide"],
            "column2": [
                "[-2] trifft nicht zu",
                "[-4] Unzulaessige Mehrfachantwort",
                "valid data",
            ],
        },
    )
    actual = _clean_invalid_data(data)
    pd.testing.assert_frame_equal(actual, expected)


def test_extract_number_from_brackets_typical_result():
    expected = pd.DataFrame({0: [123, NA, 456]}, dtype=pd.UInt16Dtype())
    data = pd.Series(["[123] valid data", "invalid data", "[456] valid data"])
    actual = _extract_number_from_brackets(data)
    assert actual.equals(expected)


def test_clean_binary_data_typical_result():
    expected = pd.Series([1, 0, 1, 0, 1], dtype=pd.UInt8Dtype())
    data = pd.Series(["yes", "no", "yes", "no", "yes"])
    actual = _clean_binary_data(data, "yes")
    assert actual.equals(expected)


def test_clean_binary_data_empty_returns_empty():
    expected = pd.Series([], dtype=pd.UInt8Dtype())
    data = pd.Series([])
    actual = _clean_binary_data(data, "yes")
    assert actual.equals(expected)


def test_positive_number_only_typical_result():
    expected = pd.Series([1, None, None, 2, None])
    data = pd.Series([1, -1, 0, 2, -2])
    actual = _positive_number_only(data)
    assert actual.equals(expected)

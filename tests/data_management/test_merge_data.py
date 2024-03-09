import pandas as pd
from wellbeing_and_machine_learning.data_management.merge_data import (
    merge_household_or_individual_datasets,
    merge_individual_and_household,
)


def test_merge_household_or_individual_datasets_dtype():
    data1 = pd.DataFrame({"key": ["A", "B", "C"], "value": [1, 2, 3]})
    data2 = pd.DataFrame({"key": ["B", "C", "D"], "value": [2, 3, 4]})
    data3 = pd.DataFrame({"key": ["C", "D", "E"], "value": [3, 4, 5]})
    actual = merge_household_or_individual_datasets([data1, data2, data3], "key")
    assert isinstance(actual, pd.DataFrame)


def test_merge_household_or_individual_datasets_number_of_rows():
    expected = 5
    data1 = pd.DataFrame({"key": ["A", "B", "C"], "value": [1, 2, 3]})
    data2 = pd.DataFrame({"key": ["B", "C", "D"], "value": [2, 3, 4]})
    data3 = pd.DataFrame({"key": ["C", "D", "E"], "value": [3, 4, 5]})
    actual = merge_household_or_individual_datasets([data1, data2, data3], "key")
    assert len(actual) == expected


def test_merge_household_or_individual_datasets_check_keys():
    expected = {"A", "B", "C", "D", "E"}
    data1 = pd.DataFrame({"key": ["A", "B", "C"], "value": [1, 2, 3]})
    data2 = pd.DataFrame({"key": ["B", "C", "D"], "value": [2, 3, 4]})
    data3 = pd.DataFrame({"key": ["C", "D", "E"], "value": [3, 4, 5]})
    actual = merge_household_or_individual_datasets([data1, data2, data3], "key")
    assert set(actual["key"]) == expected


def test_merge_individual_and_household_check_hid():
    expected = {"H2", "H3", "H4"}
    household_df = pd.DataFrame(
        {
            "hid": ["H1", "H2", "H3"],
            "syear": [2001, 2002, 2003],
            "hvalue": [10, 20, 30],
        },
    )
    individual_df = pd.DataFrame(
        {"hid": ["H2", "H3", "H4"], "syear": [2002, 2003, 2004], "ivalue": [2, 3, 4]},
    )
    actual = merge_individual_and_household(household_df, individual_df)
    assert set(actual["hid"]) == expected

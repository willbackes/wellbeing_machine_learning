import pandas as pd

from wellbeing_and_machine_learning.config import BLD, CATEGORICAL, CONTINUOUS
from wellbeing_and_machine_learning.data_management.clean_data import (
    clean_data,
    convert_categorical_to_dummy,
    observed_means_for_missing_values,
)


def task_clean_data(
    depends_on=BLD / "data" / "merge_individual_and_household.pkl",
    produces=BLD / "data" / "clean_data.pkl",
):
    merged_data = pd.read_pickle(depends_on)
    valid_data = clean_data(merged_data)
    valid_data.to_pickle(produces)


def task_convert_categorical_to_dummy(
    depends_on=BLD / "data" / "clean_data.pkl",
    produces=BLD / "data" / "clean_data_dummy.pkl",
):
    data = pd.read_pickle(depends_on)
    convert_data = convert_categorical_to_dummy(data, columns=CATEGORICAL)
    convert_data.to_pickle(produces)


def task_observed_means_for_missing_values(
    depends_on=BLD / "data" / "clean_data_dummy.pkl",
    produces=BLD / "data" / "clean_final.pkl",
):
    data = pd.read_pickle(depends_on)
    convert_data = observed_means_for_missing_values(data, columns=CONTINUOUS)
    convert_data.to_pickle(produces)

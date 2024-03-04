import pandas as pd

from wellbeing_and_machine_learning.config import BINARY, BLD, CATEGORICAL, CONTINUOUS
from wellbeing_and_machine_learning.data_management.clean_data import (
    clean_data,
    convert_categorical_to_dummy,
    observed_means_for_missing_values,
)


def task_clean_data(
    depends_on=BLD / "data" / "individual_and_household_raw.pkl",
    produces=BLD / "data" / "clean_data.pkl",
):
    merged_data = pd.read_pickle(depends_on)
    valid_data = clean_data(merged_data)
    valid_data.to_pickle(produces)


def task_convert_categorical_to_dummy(
    depends_on=BLD / "data" / "clean_data.pkl",
    produces=BLD / "data" / "dummy_variables.pkl",
):
    data = pd.read_pickle(depends_on)
    convert_data = convert_categorical_to_dummy(data, columns=CATEGORICAL)
    convert_data.to_pickle(produces)


def task_observed_means_for_missing_values(
    depends_on=BLD / "data" / "clean_data.pkl",
    produces=BLD / "data" / "continuous_variables.pkl",
):
    data = pd.read_pickle(depends_on)
    convert_data = observed_means_for_missing_values(data, columns=CONTINUOUS)
    convert_data.to_pickle(produces)


def task_clean_data_converted(
    clean_data=BLD / "data" / "clean_data.pkl",
    dummy_data=BLD / "data" / "dummy_variables.pkl",
    continuous_data=BLD / "data" / "continuous_variables.pkl",
    produces=BLD / "data" / "clean_data_converted.pkl",
):
    clean = pd.read_pickle(clean_data)
    dummy = pd.read_pickle(dummy_data)
    continuous = pd.read_pickle(continuous_data)
    data = pd.concat([clean["syear"], clean[BINARY], dummy, continuous], axis=1)
    df = data.dropna()
    df.to_pickle(produces)

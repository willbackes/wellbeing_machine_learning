import pandas as pd

from wellbeing_and_machine_learning.config import BLD, CATEGORICAL
from wellbeing_and_machine_learning.data_management.clean_data import (
    clean_categorical_to_dummy,
    clean_data,
)


def task_clean_data(
    depends_on=BLD / "data" / "merge_individual_and_household.pkl",
    produces=BLD / "data" / "clean_data.pkl",
):
    merged_data = pd.read_pickle(depends_on)
    valid_data = clean_data(merged_data)
    valid_data.to_pickle(produces)


def task_clean_categorical_to_dummy(
    depends_on=BLD / "data" / "clean_data.pkl",
    produces=BLD / "data" / "clean_data_dummy.pkl",
):
    data = pd.read_pickle(depends_on)
    valid_data = clean_categorical_to_dummy(data, columns=CATEGORICAL)
    valid_data.to_pickle(produces)

import pandas as pd

from wellbeing_and_machine_learning.config import BLD, COLS, DATA
from wellbeing_and_machine_learning.data_management.merge_data import (
    merge_household_data,
    merge_individual_and_household,
    merge_individual_data,
)

household_data_dependency = {
    data: BLD / "data" / "unzip" / f"{data}.dta" for data in DATA["household"]
}

individual_data_dependency = {
    data: BLD / "data" / "unzip" / f"{data}.dta" for data in DATA["individual"]
}


def task_merge_household_data(
    depends_on=household_data_dependency,
    produces=BLD / "data" / "household_data.pkl",
):
    household_data = [
        pd.read_stata(depends_on[data], columns=COLS[data])
        for data in DATA["household"]
    ]
    merge_household_df = merge_household_data(household_data)
    merge_household_df.to_pickle(produces)


def task_merge_individual_data(
    depends_on=individual_data_dependency,
    produces=BLD / "data" / "individual_data.pkl",
):
    individual_data = [
        pd.read_stata(depends_on[data], columns=COLS[data])
        for data in DATA["individual"]
    ]
    merge_individual_df = merge_individual_data(individual_data)
    merge_individual_df.to_pickle(produces)


def task_merge_individual_and_household(
    household_data=BLD / "data" / "household_data.pkl",
    individual_data=BLD / "data" / "individual_data.pkl",
    produces=BLD / "data" / "merge_individual_and_household.pkl",
):
    household_df = pd.read_pickle(household_data)
    individual_df = pd.read_pickle(individual_data)
    merge_individual_and_household_df = merge_individual_and_household(
        household_df,
        individual_df,
    )
    merge_individual_and_household_df.to_pickle(produces)

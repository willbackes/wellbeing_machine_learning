import pandas as pd

from wellbeing_and_machine_learning.config import BLD, COLS, DATA
from wellbeing_and_machine_learning.data_management.merge_data import (
    merge_household_or_individual_datasets,
    merge_individual_and_household,
)

household_data_dependency = {
    data: BLD / "data" / "unzip" / f"{data}.dta" for data in DATA["household"]
}

individual_data_dependency = {
    data: BLD / "data" / "unzip" / f"{data}.dta" for data in DATA["individual"]
}


def task_merge_household_data(
    depends_on=household_data_dependency,
    produces=BLD / "data" / "household_raw.pkl",
):
    household_data = [
        pd.read_stata(depends_on[data], columns=COLS[data])
        for data in DATA["household"]
    ]
    merge_household_df = merge_household_or_individual_datasets(
        household_data,
        on=["hid", "syear"],
    )
    merge_household_df.to_pickle(produces)


def task_merge_individual_data(
    depends_on=individual_data_dependency,
    health_data=BLD / "data" / "unzip" / "health.dta",
    produces=BLD / "data" / "individual_raw.pkl",
):
    individual_data = [
        pd.read_stata(depends_on[data], columns=COLS[data])
        for data in DATA["individual"]
    ]
    health_df = pd.read_stata(health_data, columns=COLS["health"])
    merge_individual_df = merge_household_or_individual_datasets(
        individual_data,
        on=["pid", "hid", "syear"],
    )
    result_df = pd.merge(
        merge_individual_df,
        health_df,
        on=["pid", "syear"],
        how="left",
    )
    result_df.to_pickle(produces)


def task_merge_individual_and_household(
    household_data=BLD / "data" / "household_raw.pkl",
    individual_data=BLD / "data" / "individual_raw.pkl",
    produces=BLD / "data" / "individual_and_household_raw.pkl",
):
    household_df = pd.read_pickle(household_data)
    individual_df = pd.read_pickle(individual_data)
    merge_individual_and_household_df = merge_individual_and_household(
        household_df,
        individual_df,
    )
    merge_individual_and_household_df.to_pickle(produces)

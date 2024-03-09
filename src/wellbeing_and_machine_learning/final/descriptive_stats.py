"""Descriptive statistics for continuous variables and permutation importance table."""

import numpy as np
import pandas as pd

from wellbeing_and_machine_learning.config import CONTINUOUS


def descriptive_stats_continuous(data):
    """Computes descriptive statistics for continuous variables.

    This function takes a pandas DataFrame, selects the continuous variables, renames them for
    readability, drops the 'agesquared' column, and computes descriptive statistics (mean, standard
    deviation, minimum, and maximum). The results are rounded to two decimal places.

    Args:
        data (pd.DataFrame): A pandas DataFrame with continuous variables.

    Returns:
        pd.DataFrame: A DataFrame with the mean, standard deviation, minimum, and maximum for each
        continuous variable.

    """
    data = data[CONTINUOUS]

    rename = {
        "age": "Age",
        "bmi": "BMI",
        "education": "Education",
        "logincome": "Log HH Income",
        "health": "Health",
        "workinghours": "Working hours",
        "numberofpeople": "Number of people in HH",
        "numberofchildren": "Number of children in HH",
        "lifesatisfaction": "Life satisfaction",
    }

    data = data.rename(columns=rename)
    data = data.drop(columns=["agesquared"])

    return np.round(data.describe(), 2).T[["mean", "std", "min", "max"]]


def permutation_importance_table(data):
    """Creates a table of permutation importance for different variables.

    This function takes a dictionary of pandas DataFrames, combines the top 10 rows of each DataFrame
    into a single DataFrame, renames the variable names for readability, and returns the combined
    DataFrame with renamed columns.

    Args:
        data (dict of pd.DataFrame): A dictionary where keys are algorithm names and values are
        pandas DataFrames. Each DataFrame should have a 'variable_name' column.

    Returns:
        pd.DataFrame: A DataFrame with renamed variable names and columns.

    """
    combined_df = pd.concat(
        [df.head(10).assign(key=key) for key, df in data.items()],
        ignore_index=True,
    )

    rename = {
        "age": "Age",
        "agesquared": "Age squared",
        "education": "Education",
        "logincome": "Log HH Income",
        "health": "Health",
        "numberofpeople": "Number of people in HH",
        "maritalstatus_[1] Verheiratet, mit Ehepartner zusammenlebend": "Marital status: married",
        "housingstatus_[1] Hauptmieter": "Housing status: main tenant",
        "housingstatus_[3] Eigentuemer": "Housing status: owner",
        "disability": "Disability status",
        "labourstatus": "Labour-force status",
    }

    combined_df["variable_name"] = combined_df["variable_name"].replace(rename)
    return combined_df.rename(
        columns={"variable_name": "Variable", "key": "Algorithm"},
    )

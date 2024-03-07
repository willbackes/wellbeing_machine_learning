import numpy as np

from wellbeing_and_machine_learning.config import CONTINUOUS


def descriptive_stats_continuous(data):
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

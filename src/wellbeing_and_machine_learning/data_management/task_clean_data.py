import pandas as pd

from wellbeing_and_machine_learning.config import BLD
from wellbeing_and_machine_learning.data_management import clean_data


def task_clean_data(
    depends_on=BLD / "data" / "merge_individual_and_household.pkl",
    produces=BLD / "data" / "clean_data.pkl",
):
    merged_data = pd.read_pickle(depends_on)
    valid_data = clean_data(merged_data)
    valid_data.to_pickle(produces)

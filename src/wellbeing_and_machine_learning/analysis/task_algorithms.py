"""Tasks for computing algorithm performance and variable importance."""

import pandas as pd
import pytask

from wellbeing_and_machine_learning.analysis.algorithms import (
    algo_performance_and_variable_importance,
    algo_performance_by_year,
)
from wellbeing_and_machine_learning.config import ALGORITHMS, BLD

for algorithm in ALGORITHMS:

    @pytask.task(id=algorithm)
    def task_algo_performance_by_year(
        depends_on=BLD / "data" / "clean_data_converted.pkl",
        algo=algorithm,
        produces=BLD / "analysis" / f"{algorithm}_performance_by_year.pkl",
    ):
        """Compute R-squared for some years for a given algorithm."""
        data = pd.read_pickle(depends_on)
        r_squared = algo_performance_by_year(data, algo)
        r_squared.to_pickle(produces)


for algorithm in ALGORITHMS:
    products = {
        "r_squared": BLD / "analysis" / f"{algorithm}_r_squared.pkl",
        "permutation_importance": BLD
        / "analysis"
        / f"{algorithm}_variable_importance.pkl",
        "prediction_df": BLD / "analysis" / f"{algorithm}_predicted_data.pkl",
    }

    @pytask.task(id=algorithm)
    def task_algo_performance_and_variable_importance(
        depends_on=BLD / "data" / "clean_data_converted.pkl",
        algo=algorithm,
        produces=products,
    ):
        """Compute R-squared and variable importance for a given algorithm."""
        data = pd.read_pickle(depends_on)
        reg_results = algo_performance_and_variable_importance(data, algo)
        for key, value in reg_results.items():
            value.to_pickle(produces[key])

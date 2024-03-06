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
        data = pd.read_pickle(depends_on)
        r_squared = algo_performance_by_year(data, algo)
        r_squared.to_pickle(produces)


for algorithm in ALGORITHMS:

    @pytask.task(id=algorithm)
    def task_algo_performance_and_variable_importance(
        depends_on=BLD / "data" / "clean_data_converted.pkl",
        algo=algorithm,
        r_squared=BLD / "analysis" / f"{algorithm}_r_squared.pkl",
        variable_importance=BLD / "analysis" / f"{algorithm}_variable_importance.pkl",
        predicted_data=BLD / "analysis" / f"{algorithm}_predicted_data.pkl",
    ):
        data = pd.read_pickle(depends_on)
        reg_results = algo_performance_and_variable_importance(data, algo)
        reg_results["r_squared"].to_pickle(r_squared)
        reg_results["permutation_importance"].to_pickle(variable_importance)
        reg_results["prediction_df"].to_pickle(predicted_data)

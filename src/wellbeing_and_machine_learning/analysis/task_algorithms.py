import pandas as pd
import pytask

from wellbeing_and_machine_learning.analysis.algorithms import (
    algo_performance_by_year,
)
from wellbeing_and_machine_learning.config import ALGORITHMS, BLD

for algo in ALGORITHMS:

    @pytask.task(id=algo)
    def task_algo_performance_by_year(
        depends_on=BLD / "data" / "clean_data_converted.pkl",
        algorithm=algo,
        produces=BLD / "analysis" / f"{algo}_performance_by_year.pkl",
    ):
        data = pd.read_pickle(depends_on)
        r_squared = algo_performance_by_year(data, algorithm)
        r_squared.to_pickle(produces)

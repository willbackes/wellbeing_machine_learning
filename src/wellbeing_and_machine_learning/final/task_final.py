import pandas as pd

from wellbeing_and_machine_learning.config import ALGORITHMS, BLD
from wellbeing_and_machine_learning.final.plot import (
    plot_average_wellbeing_by_age,
    plot_average_wellbeing_by_income,
    plot_histogram_life_satisfaction,
    plot_r_squared,
)

predicted_data_dependencies = {
    algo: BLD / "analysis" / f"{algo}_predicted_data.pkl" for algo in ALGORITHMS
}
r_squared_by_year_dependencies = {
    algo: BLD / "analysis" / f"{algo}_performance_by_year.pkl" for algo in ALGORITHMS
}


def task_plot_average_wellbeing_by_income(
    depends_on=predicted_data_dependencies,
    produces=BLD / "final" / "average_wellbeing_by_income.png",
):
    data = {algo: pd.read_pickle(val) for algo, val in depends_on.items()}
    fig = plot_average_wellbeing_by_income(data)
    fig.write_image(produces)


def task_plot_average_wellbeing_by_age(
    depends_on=predicted_data_dependencies,
    produces=BLD / "final" / "average_wellbeing_by_age.png",
):
    data = {algo: pd.read_pickle(val) for algo, val in depends_on.items()}
    fig = plot_average_wellbeing_by_age(data)
    fig.write_image(produces)


def task_histogram_life_satisfaction(
    depends_on=BLD / "data" / "clean_data_converted.pkl",
    produces=BLD / "final" / "histogram_life_satisfaction.png",
):
    data = pd.read_pickle(depends_on)
    fig = plot_histogram_life_satisfaction(data)
    fig.write_image(produces)


def task_plot_r_squared(
    depends_on=r_squared_by_year_dependencies,
    produces=BLD / "final" / "r_squared_algorithms.png",
):
    data = {algo: pd.read_pickle(val) for algo, val in depends_on.items()}
    fig = plot_r_squared(data)
    fig.write_image(produces)

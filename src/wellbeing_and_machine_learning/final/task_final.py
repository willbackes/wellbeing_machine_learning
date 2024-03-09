"""Final tasks for the project."""

import pandas as pd

from wellbeing_and_machine_learning.config import ALGORITHMS, BLD
from wellbeing_and_machine_learning.final.descriptive_stats import (
    descriptive_stats_continuous,
    permutation_importance_table,
)
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
variable_importance_dependencies = {
    algo: BLD / "analysis" / f"{algo}_variable_importance.pkl" for algo in ALGORITHMS
}


def task_plot_average_wellbeing_by_income(
    depends_on=predicted_data_dependencies,
    produces=BLD / "final" / "average_wellbeing_by_income.png",
):
    """Plot average wellbeing by income."""
    data = {algo: pd.read_pickle(val) for algo, val in depends_on.items()}
    fig = plot_average_wellbeing_by_income(data)
    fig.write_image(produces)


def task_plot_average_wellbeing_by_age(
    depends_on=predicted_data_dependencies,
    produces=BLD / "final" / "average_wellbeing_by_age.png",
):
    """Plot average wellbeing by age."""
    data = {algo: pd.read_pickle(val) for algo, val in depends_on.items()}
    fig = plot_average_wellbeing_by_age(data)
    fig.write_image(produces)


def task_histogram_life_satisfaction(
    depends_on=BLD / "data" / "clean_data_converted.pkl",
    produces=BLD / "final" / "histogram_life_satisfaction.png",
):
    """Plot histogram of life satisfaction."""
    data = pd.read_pickle(depends_on)
    fig = plot_histogram_life_satisfaction(data)
    fig.write_image(produces)


def task_plot_r_squared(
    depends_on=r_squared_by_year_dependencies,
    produces=BLD / "final" / "r_squared_algorithms.png",
):
    """Plot average R-squared for each algorithm."""
    data = {algo: pd.read_pickle(val) for algo, val in depends_on.items()}
    fig = plot_r_squared(data)
    fig.write_image(produces)


def task_descriptive_stats_continuous(
    depends_on=BLD / "data" / "clean_data_converted.pkl",
    produces=BLD / "final" / "descriptive_stats_continuous.csv",
):
    """Create descriptive statistics for continuous variables."""
    data = pd.read_pickle(depends_on)
    statistics = descriptive_stats_continuous(data)
    statistics.to_latex(produces)


def task_permutation_importance_table(
    depends_on=variable_importance_dependencies,
    produces=BLD / "final" / "permutation_importance_table.csv",
):
    """Create table of permutation importance."""
    data = {algo: pd.read_pickle(val) for algo, val in depends_on.items()}
    combined_df = permutation_importance_table(data)
    combined_df.to_latex(produces, index=False)

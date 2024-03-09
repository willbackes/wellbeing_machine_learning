"""Plotting functions for the final project."""

import pandas as pd
import plotly.express as px

from wellbeing_and_machine_learning.config import ALGORITHMS


def plot_average_wellbeing_by_income(data):
    """Plots the average wellbeing by income.

    This function takes a dictionary of pandas DataFrames, combines them into a single DataFrame,
    calculates the average predicted wellbeing ('Y_pred') for each income level for each algorithm,
    and plots the results. The plot is a line plot with income on the x-axis and average predicted
    wellbeing on the y-axis. Each algorithm is represented by a different color line.

    Args:
        data (dict of pd.DataFrame): A dictionary where keys are algorithm names and values are
        pandas DataFrames. Each DataFrame should have columns 'income' and 'Y_pred'.

    Returns:
        plotly.graph_objs._figure.Figure: A plotly Figure object that can be displayed or saved.

    """
    data_combined = pd.concat([df.assign(algorithm=alg) for alg, df in data.items()])
    average_Y_pred_by_income = (
        data_combined.groupby(["algorithm", "income"])["Y_pred"].mean().reset_index()
    )
    average_Y_pred_by_income = average_Y_pred_by_income[
        average_Y_pred_by_income["income"] < 30000
    ]
    fig = px.line(
        average_Y_pred_by_income,
        x="income",
        y="Y_pred",
        color="algorithm",
        title="Average Wellbeing by Income",
        labels={
            "income": "HH Income, monthly (000)",
            "Y_pred": "Wellbeing",
            "algorithm": "Algorithm",
        },
    )
    fig.update_layout(title_x=0.5)
    return fig


def plot_average_wellbeing_by_age(data):
    """Plots the average wellbeing by age.

    This function takes a dictionary of pandas DataFrames, combines them into a single DataFrame,
    calculates the average predicted wellbeing ('Y_pred') for each age for each algorithm,
    and plots the results. The plot is a line plot with age on the x-axis and average predicted
    wellbeing on the y-axis. Each algorithm is represented by a different color line.

    Args:
        data (dict of pd.DataFrame): A dictionary where keys are algorithm names and values are
        pandas DataFrames. Each DataFrame should have columns 'age' and 'Y_pred'.

    Returns:
        plotly.graph_objs._figure.Figure: A plotly Figure object that can be displayed or saved.

    """
    data_combined = pd.concat([df.assign(algorithm=alg) for alg, df in data.items()])
    average_Y_pred_by_age = (
        data_combined.groupby(["algorithm", "age"])["Y_pred"].mean().reset_index()
    )
    fig = px.line(
        average_Y_pred_by_age,
        x="age",
        y="Y_pred",
        color="algorithm",
        title="Average Wellbeing by Age",
        labels={"age": "Age", "Y_pred": "Wellbeing", "algorithm": "Algorithm"},
    )
    fig.update_layout(title_x=0.5)
    return fig


def plot_histogram_life_satisfaction(data):
    """Plots a histogram of life satisfaction scores.

    This function takes a pandas DataFrame, calculates the mean and variance of the 'lifesatisfaction'
    column, and plots a histogram of the 'lifesatisfaction' scores. The histogram has 11 bins and
    displays the y-axis as a fraction of the total. The x-axis categories are ordered from 1 to 10.

    Args:
        data (pd.DataFrame): A pandas DataFrame with a 'lifesatisfaction' column.

    Returns:
        plotly.graph_objs._figure.Figure: A plotly Figure object that can be displayed or saved.

    """
    mean_value = data["lifesatisfaction"].mean()
    variance_value = data["lifesatisfaction"].var()
    fig = px.histogram(
        data,
        x="lifesatisfaction",
        nbins=11,
        labels={
            "lifesatisfaction": "Life Satisfaction (μ = {:.3f}, σ = {:.3f})".format(
                mean_value,
                variance_value,
            ),
        },
        title="SOEP (2010-2018)",
        color_discrete_sequence=["grey"],
        opacity=0.7,
        histnorm="probability",  # Display y-axis as a fraction
        category_orders={
            "lifesatisfaction": list(range(1, 11)),
        },  # Specify the order of x-axis categories
    )

    mean_value = data["lifesatisfaction"].mean()
    variance_value = data["lifesatisfaction"].var()
    fig.update_layout(title_x=0.5)
    fig.update_yaxes(title_text="Fraction")

    return fig


def plot_r_squared(data):
    """Plots the average R-squared values for different algorithms.

    This function takes a dictionary of pandas DataFrames, combines them into a single DataFrame,
    calculates the mean and standard deviation of the 'r_squared' values for each algorithm,
    and plots the results. The plot is a bar plot with algorithms on the x-axis and average R-squared
    values on the y-axis. Error bars represent the standard deviation of the R-squared values.

    Args:
        data (dict of pd.DataFrame): A dictionary where keys are algorithm names and values are
        pandas DataFrames. Each DataFrame should have a 'r_squared' column.

    Returns:
        plotly.graph_objs._figure.Figure: A plotly Figure object that can be displayed or saved.

    """
    data_combined = pd.concat([df.assign(algorithm=algo) for algo, df in data.items()])
    average_r_squared = (
        data_combined.groupby(["algorithm"])["r_squared"].mean().reset_index()
    )
    std_r_squared = (
        data_combined.groupby(["algorithm"])["r_squared"].std().reset_index()
    )
    merged_data = pd.merge(
        average_r_squared,
        std_r_squared,
        on="algorithm",
        suffixes=("_mean", "_std"),
    )

    merged_data["error_bar"] = 1.96 * merged_data["r_squared_std"]
    fig = px.bar(
        merged_data,
        x="algorithm",
        y="r_squared_mean",
        error_y="error_bar",
        category_orders={"algorithm": ALGORITHMS},
        color_discrete_sequence=["grey"],
    )
    fig.update_layout(title_text="SOEP (avg. across 2010-2018)", title_x=0.5)
    fig.update_xaxes(
        title_text="Algorithms",
        tickmode="array",
        tickvals=list(range(len(ALGORITHMS))),
        ticktext=[
            f"{algo}<br>({mean:.3f})"
            for algo, mean in zip(
                ALGORITHMS,
                merged_data.loc[
                    merged_data["algorithm"].isin(ALGORITHMS),
                    "r_squared_mean",
                ],
            )
        ],
    )
    fig.update_yaxes(title_text="R2")

    return fig

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV


def ols_regression_by_year(data):
    res = {}
    for year in data["syear"].unique():
        yearly_data = data[data["syear"] == year]
        X = yearly_data.drop(["lifesatisfaction"], axis=1)
        Y = yearly_data["lifesatisfaction"]

        ols_model = LinearRegression()
        ols_model.fit(X, Y)
        res[year] = ols_model.score(X, Y)
    return res


def lasso_regression_by_year(data, alphas=None, cv=4):
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1, 10]
    res = {}
    for year in data["syear"].unique():
        yearly_data = data[data["syear"] == year]
        X = yearly_data.drop(["lifesatisfaction"], axis=1)
        Y = yearly_data["lifesatisfaction"]

        param_grid = {"alpha": alphas}
        lasso_model = Lasso()
        grid_search = GridSearchCV(
            estimator=lasso_model,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        grid_search.fit(X, Y)
        best_lasso_model = grid_search.best_estimator_
        r_squared = best_lasso_model.score(X, Y)
        res[year] = r_squared
    return res

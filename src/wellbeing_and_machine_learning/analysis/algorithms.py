from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


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


def random_forest_regression_by_year(
    data,
    n_estimators=100,
    max_depth=None,
    random_state=42,
):
    res = {}
    for year in data["syear"].unique():
        yearly_data = data[data["syear"] == year]
        X = yearly_data.drop(["lifesatisfaction"], axis=1)
        Y = yearly_data["lifesatisfaction"]

        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=0.2,
            random_state=42,
        )

        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        rf_model.fit(X, Y)
        rf_model.fit(X_train, Y_train)
        Y_pred = rf_model.predict(X_test)
        r_squared = r2_score(Y_test, Y_pred)
        res[year] = r_squared
    return res


def gradient_boosting_regression_by_year(
    data,
    learning_rate=0.005,
    n_estimators=100,
    max_depth=8,
    random_state=42,
):
    res = {}
    for year in data["syear"].unique():
        yearly_data = data[data["syear"] == year]
        X = yearly_data.drop(["lifesatisfaction"], axis=1)
        Y = yearly_data["lifesatisfaction"]

        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=0.2,
            random_state=42,
        )

        gb_model = GradientBoostingRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        gb_model.fit(X_train, Y_train)
        Y_pred = gb_model.predict(X_test)
        r_squared = r2_score(Y_test, Y_pred)
        res[year] = r_squared
    return res

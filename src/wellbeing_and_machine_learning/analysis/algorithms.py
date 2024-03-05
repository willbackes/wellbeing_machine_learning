from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


def algo_performance_by_year(data, algo):
    r_squared = {}

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

        if algo == "ols":
            r_squared[year] = _ols_regression(X_train, X_test, Y_train, Y_test)
        elif algo == "lasso":
            r_squared[year] = _lasso_regression(X_train, X_test, Y_train, Y_test)
        elif algo == "random_forest":
            r_squared[year] = _random_forest_regression(
                X_train,
                X_test,
                Y_train,
                Y_test,
            )
        elif algo == "gradient_boosting":
            r_squared[year] = _gradient_boosting_regression(
                X_train,
                X_test,
                Y_train,
                Y_test,
            )
        else:
            msg = f"Unknown algorithm: {algo}"
            raise ValueError(msg)
    return r_squared


def _ols_regression(X_train, X_test, Y_train, Y_test):
    ols_model = LinearRegression()
    ols_model.fit(X_train, Y_train)
    return ols_model.score(X_test, Y_test)


def _lasso_regression(X_train, X_test, Y_train, Y_test, alphas=None, cv=4):
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1, 10]

    param_grid = {"alpha": alphas}
    lasso_model = Lasso()
    grid_search = GridSearchCV(
        estimator=lasso_model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    grid_search.fit(X_train, Y_train)
    best_lasso_model = grid_search.best_estimator_
    return best_lasso_model.score(X_test, Y_test)


def _random_forest_regression(
    X_train,
    X_test,
    Y_train,
    Y_test,
    n_estimators=100,
    max_depth=None,
    random_state=42,
):
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    rf_model.fit(X_train, Y_train)
    Y_pred = rf_model.predict(X_test)
    return r2_score(Y_test, Y_pred)


def _gradient_boosting_regression(
    X_train,
    X_test,
    Y_train,
    Y_test,
    learning_rate=0.005,
    n_estimators=100,
    max_depth=8,
    random_state=42,
):
    gb_model = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    gb_model.fit(X_train, Y_train)
    Y_pred = gb_model.predict(X_test)
    return r2_score(Y_test, Y_pred)

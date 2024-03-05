import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


def algo_performance_by_year(data, algo):
    results = {"syear": [], "r_squared": []}

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
            r_squared = _ols_regression(
                X_train,
                X_test,
                Y_train,
                Y_test,
                r_squared_only=True,
            )
        elif algo == "lasso":
            r_squared = _lasso_regression(
                X_train,
                X_test,
                Y_train,
                Y_test,
                r_squared_only=True,
            )
        elif algo == "random_forest":
            r_squared = _random_forest_regression(
                X_train,
                X_test,
                Y_train,
                Y_test,
                r_squared_only=True,
            )
        elif algo == "gradient_boosting":
            r_squared = _gradient_boosting_regression(
                X_train,
                X_test,
                Y_train,
                Y_test,
                r_squared_only=True,
            )
        else:
            msg = f"Unknown algorithm: {algo}"
            raise ValueError(msg)

        results["syear"].append(year)
        results["r_squared"].append(r_squared)

    return pd.DataFrame(results)


def algo_performance_and_variable_importance(data, algo):
    results = {
        "r_squared": None,
        "permutation_importance": None,
        "prediction_df": None,
    }

    X = data.drop(["lifesatisfaction"], axis=1)
    Y = data["lifesatisfaction"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )

    if algo == "ols":
        (
            results["r_squared"],
            results["permutation_importance"],
            results["prediction_df"],
        ) = _ols_regression(X_train, X_test, Y_train, Y_test, r_squared_only=False)
    elif algo == "lasso":
        (
            results["r_squared"],
            results["permutation_importance"],
            results["prediction_df"],
        ) = _lasso_regression(X_train, X_test, Y_train, Y_test, r_squared_only=False)
    elif algo == "random_forest":
        (
            results["r_squared"],
            results["permutation_importance"],
            results["prediction_df"],
        ) = _random_forest_regression(
            X_train,
            X_test,
            Y_train,
            Y_test,
            r_squared_only=False,
        )
    elif algo == "gradient_boosting":
        (
            results["r_squared"],
            results["permutation_importance"],
            results["prediction_df"],
        ) = _gradient_boosting_regression(
            X_train,
            X_test,
            Y_train,
            Y_test,
            r_squared_only=False,
        )
    else:
        msg = f"Unknown algorithm: {algo}"
        raise ValueError(msg)

    return results


def _ols_regression(X_train, X_test, Y_train, Y_test, r_squared_only):
    ols_model = LinearRegression()
    ols_model.fit(X_train, Y_train)
    Y_pred = ols_model.predict(X_test)

    if r_squared_only is True:
        return r2_score(Y_test, Y_pred)
    else:
        r_squared = r2_score(Y_test, Y_pred)

        perm_importance_df = _variable_importance(ols_model, X_test, Y_test)

        prediction_df = pd.DataFrame(
            {"Y_pred": Y_pred, "income": X_test["logincome"], "age": X_test["age"]},
        )

        return r_squared, perm_importance_df, prediction_df


def _lasso_regression(X_train, X_test, Y_train, Y_test, r_squared_only, alphas=None):
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1, 10]

    param_grid = {"alpha": alphas}
    lasso_model = Lasso()
    grid_search = GridSearchCV(
        estimator=lasso_model,
        param_grid=param_grid,
        cv=4,
        scoring="neg_mean_squared_error",
    )
    grid_search.fit(X_train, Y_train)
    best_lasso_model = grid_search.best_estimator_
    Y_pred = best_lasso_model.predict(X_test)

    if r_squared_only is True:
        return r2_score(Y_test, Y_pred)
    else:
        r_squared = r2_score(Y_test, Y_pred)

        perm_importance_df = _variable_importance(best_lasso_model, X_test, Y_test)

        prediction_df = pd.DataFrame(
            {"Y_pred": Y_pred, "income": X_test["logincome"], "age": X_test["age"]},
        )

        return r_squared, perm_importance_df, prediction_df


def _random_forest_regression(
    X_train,
    X_test,
    Y_train,
    Y_test,
    r_squared_only,
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

    if r_squared_only is True:
        return r2_score(Y_test, Y_pred)
    else:
        r_squared = r2_score(Y_test, Y_pred)

        perm_importance_df = _variable_importance(rf_model, X_test, Y_test)

        prediction_df = pd.DataFrame(
            {"Y_pred": Y_pred, "income": X_test["logincome"], "age": X_test["age"]},
        )

        return r_squared, perm_importance_df, prediction_df


def _gradient_boosting_regression(
    X_train,
    X_test,
    Y_train,
    Y_test,
    r_squared_only,
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

    if r_squared_only is True:
        return r2_score(Y_test, Y_pred)
    else:
        r_squared = r2_score(Y_test, Y_pred)

        perm_importance_df = _variable_importance(gb_model, X_test, Y_test)

        prediction_df = pd.DataFrame(
            {"Y_pred": Y_pred, "income": X_test["logincome"], "age": X_test["age"]},
        )

        return r_squared, perm_importance_df, prediction_df


def _variable_importance(model, X_test, Y_test):
    perm_importance = permutation_importance(
        model,
        X_test,
        Y_test,
        n_repeats=30,
        random_state=42,
    )
    variable_name = list(X_test.columns)
    perm_importance_values = perm_importance.importances_mean
    perm_importance_df = pd.DataFrame(
        {"variable_name": variable_name, "PI": perm_importance_values},
    )
    return perm_importance_df.sort_values(by="PI", ascending=False)

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


def algo_performance_by_year(data, algo):
    """Computes the performance of a given algorithm by year.

    This function takes a pandas DataFrame and an algorithm name, splits the data into training and
    testing sets for each unique year, fits the specified algorithm to the training data, computes
    the R-squared value for the testing data, and returns a DataFrame with the R-squared values for
    each year.

    Args:
        data (pd.DataFrame): A pandas DataFrame with a 'syear' column and a 'lifesatisfaction' column.
        algo (str): The name of the algorithm to use. Options are 'ols', 'lasso', 'random_forest',
        and 'gradient_boosting'.

    Returns:
        pd.DataFrame: A DataFrame with 'syear' and 'r_squared' columns.

    """
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
                X,
                Y,
                r_squared_only=True,
            )
        elif algo == "lasso":
            r_squared = _lasso_regression(
                X,
                Y,
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
    """Computes the performance and variable importance of a given algorithm.

    This function takes a pandas DataFrame and an algorithm name, splits the data into training and
    testing sets, fits the specified algorithm to the training data, computes the R-squared value
    and permutation importance for the testing data, and returns a dictionary with the results.

    Args:
        data (pd.DataFrame): A pandas DataFrame with a 'lifesatisfaction' column.
        algo (str): The name of the algorithm to use. Options are 'ols', 'lasso', 'random_forest',
        and 'gradient_boosting'.

    Returns:
        dict: A dictionary with 'r_squared', 'permutation_importance', and 'prediction_df' keys.

    """
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
        ) = _ols_regression(X, Y, r_squared_only=False)
    elif algo == "lasso":
        (
            results["r_squared"],
            results["permutation_importance"],
            results["prediction_df"],
        ) = _lasso_regression(X, Y, r_squared_only=False)
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


def _ols_regression(X, Y, r_squared_only):
    """Fits an Ordinary Least Squares (OLS) regression model and computes the R-squared
    value.

    This function takes a features DataFrame (X), a target Series (Y), and a boolean flag
    (r_squared_only), fits an OLS regression model to the data, predicts the target values, and
    computes the R-squared value. If r_squared_only is True, it returns the R-squared value.
    Otherwise, it also computes the permutation importance of the variables and returns a DataFrame
    with the predicted values, the log of income, and age.

    Args:
        X (pd.DataFrame): A pandas DataFrame with the feature variables.
        Y (pd.Series): A pandas Series with the target variable.
        r_squared_only (bool): A flag indicating whether to return only the R-squared value.

    Returns:
        float or tuple: If r_squared_only is True, a float representing the R-squared value.
        Otherwise, a tuple with a DataFrame of the R-squared value, a DataFrame of the permutation
        importance, and a DataFrame with the predicted values, the log of income, and age.

    """
    ols_model = LinearRegression()
    ols_model.fit(X, Y)
    Y_pred = ols_model.predict(X)

    if r_squared_only is True:
        return r2_score(Y, Y_pred)
    else:
        r_squared = r2_score(Y, Y_pred)
        r_squared_df = pd.DataFrame([r_squared], columns=["r_squared"])

        perm_importance_df = _variable_importance(ols_model, X, Y)

        prediction_df = pd.DataFrame(
            {"Y_pred": Y_pred, "logincome": X["logincome"], "age": X["age"]},
        )
        prediction_df["income"] = np.exp(prediction_df["logincome"])

        return r_squared_df, perm_importance_df, prediction_df


def _lasso_regression(X, Y, r_squared_only, use_grid_search=True):
    """Fits a Lasso regression model and computes the R-squared value.

    This function takes a features DataFrame (X), a target Series (Y), a boolean flag
    (r_squared_only), and a boolean flag (use_grid_search), fits a Lasso regression model to the
    data, predicts the target values, and computes the R-squared value. If use_grid_search is True,
    it uses GridSearchCV to find the best alpha parameter. If r_squared_only is True, it returns
    the R-squared value. Otherwise, it also computes the permutation importance of the variables
    and returns a DataFrame with the predicted values, the log of income, and age.

    Args:
        X (pd.DataFrame): A pandas DataFrame with the feature variables.
        Y (pd.Series): A pandas Series with the target variable.
        r_squared_only (bool): A flag indicating whether to return only the R-squared value.
        use_grid_search (bool): A flag indicating whether to use GridSearchCV to find the best
        alpha parameter.

    Returns:
        float or tuple: If r_squared_only is True, a float representing the R-squared value.
        Otherwise, a tuple with a DataFrame of the R-squared value, a DataFrame of the permutation
        importance, and a DataFrame with the predicted values, the log of income, and age.

    """
    if use_grid_search:
        param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10]}
        lasso_model = Lasso()
        grid_search = GridSearchCV(
            estimator=lasso_model,
            param_grid=param_grid,
            cv=4,
            scoring="neg_mean_squared_error",
        )
        grid_search.fit(X, Y)
        best_lasso_model = grid_search.best_estimator_
    else:
        best_lasso_model = Lasso(alpha=0.001)
        best_lasso_model.fit(X, Y)

    Y_pred = best_lasso_model.predict(X)

    if r_squared_only is True:
        return r2_score(Y, Y_pred)
    else:
        r_squared = r2_score(Y, Y_pred)
        r_squared_df = pd.DataFrame([r_squared], columns=["r_squared"])

        perm_importance_df = _variable_importance(best_lasso_model, X, Y)

        prediction_df = pd.DataFrame(
            {"Y_pred": Y_pred, "logincome": X["logincome"], "age": X["age"]},
        )
        prediction_df["income"] = np.exp(prediction_df["logincome"])

        return r_squared_df, perm_importance_df, prediction_df


def _random_forest_regression(
    X_train,
    X_test,
    Y_train,
    Y_test,
    r_squared_only,
    use_grid_search=False,
    n_estimators=100,
    max_depth=None,
    random_state=42,
):
    """Fits a Random Forest regression model and computes the R-squared value.

    This function takes training and testing data, a boolean flag (r_squared_only), and parameters
    for a Random Forest model, fits the model to the training data, predicts the target values for
    the testing data, and computes the R-squared value. If use_grid_search is True, it uses
    GridSearchCV to find the best parameters. If r_squared_only is True, it returns the R-squared
    value. Otherwise, it also computes the permutation importance of the variables and returns a
    DataFrame with the predicted values, the log of income, and age.

    Args:
        X_train (pd.DataFrame): A pandas DataFrame with the training feature variables.
        X_test (pd.DataFrame): A pandas DataFrame with the testing feature variables.
        Y_train (pd.Series): A pandas Series with the training target variable.
        Y_test (pd.Series): A pandas Series with the testing target variable.
        r_squared_only (bool): A flag indicating whether to return only the R-squared value.
        use_grid_search (bool): A flag indicating whether to use GridSearchCV to find the best
        parameters.
        n_estimators (int): The number of trees in the forest.
        max_depth (int): The maximum depth of the tree.
        random_state (int): A seed used by the random number generator.

    Returns:
        float or tuple: If r_squared_only is True, a float representing the R-squared value.
        Otherwise, a tuple with a DataFrame of the R-squared value, a DataFrame of the permutation
        importance, and a DataFrame with the predicted values, the log of income, and age.

    """
    if use_grid_search:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 50, 100],
        }
        rf = RandomForestRegressor(random_state=random_state)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        best_params = grid_search.best_params_
        rf_model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=random_state,
        )
    else:
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
        r_squared_df = pd.DataFrame([r_squared], columns=["r_squared"])

        perm_importance_df = _variable_importance(rf_model, X_test, Y_test)

        prediction_df = pd.DataFrame(
            {"Y_pred": Y_pred, "logincome": X_test["logincome"], "age": X_test["age"]},
        )
        prediction_df["income"] = np.exp(prediction_df["logincome"])

        return r_squared_df, perm_importance_df, prediction_df


def _gradient_boosting_regression(
    X_train,
    X_test,
    Y_train,
    Y_test,
    r_squared_only,
    use_grid_search=False,
    learning_rate=0.005,
    n_estimators=100,
    max_depth=8,
    random_state=42,
):
    """Fits a Gradient Boosting regression model and computes the R-squared value.

    This function takes training and testing data, a boolean flag (r_squared_only), and parameters
    for a Gradient Boosting model, fits the model to the training data, predicts the target values
    for the testing data, and computes the R-squared value. If use_grid_search is True, it uses
    GridSearchCV to find the best parameters. If r_squared_only is True, it returns the R-squared
    value. Otherwise, it also computes the permutation importance of the variables and returns a
    DataFrame with the predicted values, the log of income, and age.

    Args:
        X_train (pd.DataFrame): A pandas DataFrame with the training feature variables.
        X_test (pd.DataFrame): A pandas DataFrame with the testing feature variables.
        Y_train (pd.Series): A pandas Series with the training target variable.
        Y_test (pd.Series): A pandas Series with the testing target variable.
        r_squared_only (bool): A flag indicating whether to return only the R-squared value.
        use_grid_search (bool): A flag indicating whether to use GridSearchCV to find the best
        parameters.
        learning_rate (float): The learning rate shrinks the contribution of each tree.
        n_estimators (int): The number of boosting stages to perform.
        max_depth (int): The maximum depth of the individual regression estimators.
        random_state (int): A seed used by the random number generator.

    Returns:
        float or tuple: If r_squared_only is True, a float representing the R-squared value.
        Otherwise, a tuple with a DataFrame of the R-squared value, a DataFrame of the permutation
        importance, and a DataFrame with the predicted values, the log of income, and age.

    """
    if use_grid_search:
        # Define the parameter grid
        param_grid = {
            "learning_rate": [0.001, 0.005, 0.01],
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 8],
        }
        gb = GradientBoostingRegressor(random_state=random_state)
        grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        best_params = grid_search.best_params_
        gb_model = GradientBoostingRegressor(
            learning_rate=best_params["learning_rate"],
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=random_state,
        )
    else:
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
        r_squared_df = pd.DataFrame([r_squared], columns=["r_squared"])

        perm_importance_df = _variable_importance(gb_model, X_test, Y_test)

        prediction_df = pd.DataFrame(
            {"Y_pred": Y_pred, "logincome": X_test["logincome"], "age": X_test["age"]},
        )
        prediction_df["income"] = np.exp(prediction_df["logincome"])

        return r_squared_df, perm_importance_df, prediction_df


def _variable_importance(model, X_test, Y_test):
    """Computes the permutation importance of the variables.

    This function takes a fitted model and testing data, computes the permutation importance of the
    variables using the model and the testing data, and returns a DataFrame with the variable names
    and their permutation importance values, sorted in descending order of importance.

    Args:
        model (sklearn estimator): A fitted scikit-learn estimator.
        X_test (pd.DataFrame): A pandas DataFrame with the testing feature variables.
        Y_test (pd.Series): A pandas Series with the testing target variable.

    Returns:
        pd.DataFrame: A DataFrame with the variable names and their permutation importance values,
        sorted in descending order of importance.

    """
    perm_importance = permutation_importance(
        model,
        X_test,
        Y_test,
        n_repeats=10,
        random_state=42,
    )
    variable_name = list(X_test.columns)
    perm_importance_values = perm_importance.importances_mean
    perm_importance_df = pd.DataFrame(
        {"variable_name": variable_name, "PI": perm_importance_values},
    )
    return perm_importance_df.sort_values(by="PI", ascending=False)

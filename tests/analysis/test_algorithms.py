import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from wellbeing_and_machine_learning.analysis.algorithms import (
    _gradient_boosting_regression,
    _lasso_regression,
    _ols_regression,
    _random_forest_regression,
    _variable_importance,
)


def test_ols_regression_returns_float():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    actual = _ols_regression(X, Y, r_squared_only=True)
    assert isinstance(actual, float)


def test_ols_regression_result_within_range():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    actual = _ols_regression(X, Y, r_squared_only=True)
    assert 0 <= actual <= 1


def test_ols_regression_full_output_length():
    expected = 3
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    actual = _ols_regression(X, Y, r_squared_only=False)
    assert len(actual) == expected


def test_ols_regression_full_output_r_squared_df():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    r_squared_df, _, _ = _ols_regression(X, Y, r_squared_only=False)
    assert "r_squared" in r_squared_df.columns


def test_ols_regression_full_output_prediction_df():
    expected = {"Y_pred", "logincome", "age", "income"}
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    _, _, prediction_df = _ols_regression(X, Y, r_squared_only=False)
    assert set(prediction_df.columns) == expected


def test_lasso_regression_full_output_length():
    expected = 3
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    actual = _lasso_regression(X, Y, r_squared_only=False)
    assert len(actual) == expected


def test_lasso_regression_full_output_r_squared_df():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    r_squared_df, _, _ = _lasso_regression(X, Y, r_squared_only=False)
    assert "r_squared" in r_squared_df.columns


def test_lasso_regression_full_output_prediction_df():
    expected = {"Y_pred", "logincome", "age", "income"}
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    _, _, prediction_df = _lasso_regression(X, Y, r_squared_only=False)
    assert set(prediction_df.columns) == expected


def test_lasso_regression_with_grid_search():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    actual = _lasso_regression(X, Y, r_squared_only=True, use_grid_search=True)
    assert 0 <= actual <= 1


def test_lasso_regression_without_grid_search():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    actual = _lasso_regression(X, Y, r_squared_only=True, use_grid_search=False)
    assert 0 <= actual <= 1


def test_random_forest_regression_r_squared_only():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )

    actual = _random_forest_regression(
        X_train,
        X_test,
        Y_train,
        Y_test,
        r_squared_only=True,
    )

    assert isinstance(actual, float)
    assert 0 <= actual <= 1


def test_random_forest_regression_full_output_length():
    expected = 3
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    actual = _random_forest_regression(
        X_train,
        X_test,
        Y_train,
        Y_test,
        r_squared_only=False,
    )
    assert len(actual) == expected


def test_random_forest_regression_full_output_r_squared_df():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    r_squared_df, _, _ = _random_forest_regression(
        X_train,
        X_test,
        Y_train,
        Y_test,
        r_squared_only=False,
    )
    assert "r_squared" in r_squared_df.columns


def test_random_forest_regression_full_output_perm_importance_df():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    _, perm_importance_df, _ = _random_forest_regression(
        X_train,
        X_test,
        Y_train,
        Y_test,
        r_squared_only=False,
    )
    assert isinstance(perm_importance_df, pd.DataFrame)


def test_random_forest_regression_full_output_prediction_df():
    expected = {"Y_pred", "logincome", "age", "income"}
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    _, _, prediction_df = _random_forest_regression(
        X_train,
        X_test,
        Y_train,
        Y_test,
        r_squared_only=False,
    )
    assert set(prediction_df.columns) == expected


def test_gradient_boosting_regression_r_squared_only():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    actual = _gradient_boosting_regression(
        X_train,
        X_test,
        Y_train,
        Y_test,
        r_squared_only=True,
    )
    assert 0 <= actual <= 1


def test_gradient_boosting_regression_full_output_length():
    expected = 3
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    actual = _gradient_boosting_regression(
        X_train,
        X_test,
        Y_train,
        Y_test,
        r_squared_only=False,
    )
    assert len(actual) == expected


def test_gradient_boosting_regression_full_output_r_squared_df():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    r_squared_df, _, _ = _gradient_boosting_regression(
        X_train,
        X_test,
        Y_train,
        Y_test,
        r_squared_only=False,
    )
    assert "r_squared" in r_squared_df.columns


def test_gradient_boosting_regression_full_output_perm_importance_df():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    _, perm_importance_df, _ = _gradient_boosting_regression(
        X_train,
        X_test,
        Y_train,
        Y_test,
        r_squared_only=False,
    )
    assert isinstance(perm_importance_df, pd.DataFrame)


def test_gradient_boosting_regression_full_output_prediction_df():
    expected = {"Y_pred", "logincome", "age", "income"}
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    _, _, prediction_df = _gradient_boosting_regression(
        X_train,
        X_test,
        Y_train,
        Y_test,
        r_squared_only=False,
    )
    assert set(prediction_df.columns) == expected


def test_variable_importance_output_type():
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    model = LinearRegression().fit(X_train, Y_train)
    actual = _variable_importance(model, X_test, Y_test)
    assert isinstance(actual, pd.DataFrame)


def test_variable_importance_output_columns():
    expected = {"variable_name", "PI"}
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    model = LinearRegression().fit(X_train, Y_train)
    actual = _variable_importance(model, X_test, Y_test)
    assert set(actual.columns) == expected


def test_variable_importance_output_values():
    expected = 0
    X, Y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X = pd.DataFrame(X, columns=["logincome", "age"])
    Y = pd.Series(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression().fit(X_train, Y_train)
    actual = _variable_importance(model, X_test, Y_test)

    assert all(actual["PI"] >= expected)

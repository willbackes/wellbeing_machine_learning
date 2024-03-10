import unittest

import numpy as np
import pandas as pd

# Assuming the _ols_regression and _variable_importance functions are in a module named 'algorithms'
from wellbeing_and_machine_learning.analysis.algorithms import (
    _ols_regression,
)


class TestOLSRegression(unittest.TestCase):
    def setUp(self):
        # Create some dummy data for testing
        self.X = pd.DataFrame(
            {"logincome": np.random.rand(10), "age": np.random.randint(20, 50, 10)},
        )
        self.Y = pd.Series(np.random.rand(10))
        self.r_squared_only = False

    def test_ols_regression_r_squared_only(self):
        # Test the function when r_squared_only is True
        r_squared = _ols_regression(self.X, self.Y, True)
        assert isinstance(r_squared, float)

    def test_ols_regression_full_output(self):
        # Test the function when r_squared_only is False
        r_squared_df, perm_importance_df, prediction_df = _ols_regression(
            self.X,
            self.Y,
            self.r_squared_only,
        )

        # Check the types of the outputs
        assert isinstance(r_squared_df, pd.DataFrame)
        assert isinstance(perm_importance_df, pd.DataFrame)
        assert isinstance(prediction_df, pd.DataFrame)

        # Check the shapes of the outputs
        assert r_squared_df.shape == (1, 1)
        assert prediction_df.shape == (10, 4)

        # Check the columns of the outputs
        assert list(r_squared_df.columns) == ["r_squared"]
        assert "Y_pred" in prediction_df.columns
        assert "logincome" in prediction_df.columns
        assert "age" in prediction_df.columns
        assert "income" in prediction_df.columns


if __name__ == "__main__":
    unittest.main()

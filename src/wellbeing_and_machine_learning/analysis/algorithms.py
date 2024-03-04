from sklearn.linear_model import LinearRegression


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

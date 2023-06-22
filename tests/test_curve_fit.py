import pandas as pd
from scipy.optimize import curve_fit
import numpy as np


def curve_func(x, a, k):
    return a * np.exp(-k * x)


def linear_func(x, a, b):
    return a * x + b


def test_scipy():
    df = pd.read_csv("/Users/juusoahlroos/Documents/own/gradu/tests/test_csv.csv")
    x_data = df.price
    y_data = df.level_size
    param, _ = curve_fit(
        f=curve_func,
        xdata=x_data,
        ydata=y_data,
        p0=(10000, 100),
        method="dogbox",
        bounds=([0, 0], [np.inf, np.inf]),
    )
    a, k = param
    print(f"a: {a}, k: {k}")


def test_linear_transform():
    df = pd.read_csv("/Users/juusoahlroos/Documents/own/gradu/tests/test_csv.csv")
    ln_l = np.log(df.level_size)
    p = df.price

    covariance = np.cov(p, ln_l)
    var_p = covariance[0, 0]

    slope = covariance[0, 1] / var_p
    intercept = np.mean(ln_l) - slope * np.mean(p)

    k = -slope
    a = np.exp(intercept)
    print(f"a: {a}, k: {k}")


# import statsmodels.api as sm


# def test_ols():
#     df = pd.read_csv("/Users/juusoahlroos/Documents/own/gradu/tests/test_csv.csv")
#     ln_l = np.log(df.level_size)
#     p = df.price

#     # Add constant (intercept term) to x.
#     p = sm.add_constant(p)

#     # Fit the model
#     model = sm.OLS(ln_l, p)
#     results = model.fit()

#     # The parameters (intercept and slope) are contained in results.params
#     intercept, slope = results.params

#     k = -slope
#     a = np.exp(intercept)

#     print(f"a: {a}, k: {k}")

#     intercept_2, slope2 = sm.OLS(p, ln_l).fit().params
#     print(f"a: {np.exp(intercept_2)}, k: {-slope2}")

import pandas as pd
import numpy as np
import statsmodels.api as sm


def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    filter = (df[column_name] >= Q1 - 1.5 * IQR) & (df[column_name] <= Q3 + 1.5 * IQR)
    return df.loc[filter]


def test_linear_transform_cut_outliers():
    df = pd.read_csv("/Users/juusoahlroos/Documents/own/gradu/tests/test_csv.csv")

    df = df.sort_values(by="price").reset_index(drop=True)
    df = df.iloc[:12]

    ln_l = np.log(df.level_size)
    p = df.price

    # Add constant (intercept term) to x
    p = sm.add_constant(p)

    # Fit the model
    model = sm.OLS(ln_l, p)
    results = model.fit()

    # The parameters (intercept and slope) are contained in results.params
    intercept, slope = results.params

    k = -slope
    a = np.exp(intercept)

    print(f"a: {a}, k: {k}")


def test_curve_fit_linear_func():
    df = pd.read_csv("/Users/juusoahlroos/Documents/own/gradu/tests/test_csv.csv")
    x = df.price
    y = np.log(df.level_size)

    param, _ = curve_fit(
        f=linear_func,
        xdata=x,
        ydata=y,
        p0=(10, 10),
        method="dogbox",
        bounds=([0, 0], [np.inf, np.inf]),
    )
    print(param)


# def test_log_estimation():
#     df = pd.read_csv("/Users/juusoahlroos/Documents/own/gradu/tests/test_csv.csv")

#     log_x = np.log(df.price)
#     log_y = np.log(df.level_size)

#     cov = np.cov(log_x, log_y)
#     var_x = cov[0, 0]
#     slope = cov[0, 1] / var_x
#     intercept = np.mean(log_y) - slope * np.mean(log_x)

#     print(f"slope: {slope}, intercept: {intercept}")
#     print(f"a: {np.exp(intercept)}, k: {-slope}")


# def test_method_of_moments():
#     df = pd.read_csv("/Users/juusoahlroos/Documents/own/gradu/tests/test_csv.csv")
#     x_data = df.price
#     y_data = df.level_size

#     # First moment (mean)
#     mean_y = np.mean(y_data)

#     # Second moment (variance)
#     var_y = np.var(y_data)

#     # From the moments of an exponential distribution,
#     # mean = a/k
#     # variance = a/(k^2)
#     # We can then solve these to get estimates for a and k.
#     k_est = mean_y / var_y
#     a_est = mean_y**2 / var_y

#     print(f"a: {a_est}, k: {k_est}")

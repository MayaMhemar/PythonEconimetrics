from colorama import Fore
import numpy as np
from  sklearn.linear_model import LinearRegression
import statsmodels.api as sm

print(f"Варіант 13")

def adv_regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())

    r_sq = model.score(x, y)
    print(f"coefficient of determination: {r_sq}")
    print(f"intercept b0: {model.intercept_}")
    print(f"coefficients: {model.coef_}")

    y_pred = model.predict(x)
    print(f"predicted response: \n{y_pred}")

def adv_regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    print(Fore.LIGHTCYAN_EX,f"\n{results.summary()}")

x = [
    [41.8, 7.2, 35.5], [44.6, 9.2, 34.4], [42, 11.6, 30.5],
    [49.3, 13.4, 36.5], [48.6, 13.4, 32.1], [54.2, 14.8, 40.1],
    [62.2, 15.4, 34.1], [57.3, 15.8, 39.3], [54.1, 16.2, 39.2],
    [69.4, 16.5, 41.1], [60.2, 17, 42.5], [65.2, 17.1, 45.2],
    [70.1, 18, 45.8], [75.5, 18.5, 43.9], [74.9, 19, 50.5], [72.3, 20.5, 48.3]]

y = [4.5, 5.15, 6.00, 5.55, 5.70, 6.55, 5.90, 6.15, 6.95, 6.40, 6.90, 7.35, 7.80, 8.00, 8.80, 9.30]

adv_regression(x, y)
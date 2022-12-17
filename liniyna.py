from colorama import Fore
import numpy as np
from sklearn.linear_model import LinearRegression

print(f"Варіант 13")

print(f"Подамо x як двовимірний масив:")
x = np.array([4.2, 4.8, 5.2, 7.2, 8.9, 9.2, 10.1, 11.9, 11.9, 14.8]).reshape((-1, 1))
y = np.array([17.3, 18.6, 20.3, 22.8, 23.8, 25.8, 27.5, 28.7, 29.1, 36.2])

model = LinearRegression()
print(model.fit(x, y))

r_sq = round(model.score(x, y), 4)
print(f"Коефіцієнт детермінації (𝑅²): {r_sq}")

print(f"intercept (a, 𝑏₀): {round(model.intercept_, 4)}")
print(f"slope (b, 𝑏₁): {model.coef_}")

print(f"Подамо і y як двовимірний масив:")

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept (a, 𝑏₀): {new_model.intercept_}")
print(f"slope (b, 𝑏₁): {new_model.coef_}")

y_pred = model.predict(x)
print(f"predicted response (g(xi)): \n{y_pred}")

y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response (g(xi)):\n{y_pred}")

x_new = np.arange(5).reshape((-1, 1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

print(Fore.LIGHTCYAN_EX + f"Висновки:"
                 f"\n (𝑅²): {r_sq},"
                 f"\n (a, 𝑏₀): {round(model.intercept_, 4)},"
                 f"\n (b, 𝑏₁): {model.coef_}")
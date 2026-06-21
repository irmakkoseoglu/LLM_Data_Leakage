import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

df = pd.read_csv("house_prices.csv")
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
best_alpha = None
best_rmse = float("inf")

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print(f"alpha={alpha}: RMSE={rmse:.2f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha}")
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train, y_train)

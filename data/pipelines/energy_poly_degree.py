import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

df = pd.read_csv("energy_consumption.csv")
X = df.drop("consumption", axis=1)
y = df["consumption"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

best_degree = 1
best_r2 = -np.inf

for degree in [1, 2, 3, 4]:
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("lr",   LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    print(f"degree={degree}: R2={r2:.3f}")
    if r2 > best_r2:
        best_r2 = r2
        best_degree = degree

print(f"\nBest polynomial degree: {best_degree}")

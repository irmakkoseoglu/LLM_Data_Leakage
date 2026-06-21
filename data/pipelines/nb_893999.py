import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Temporal leakage: future average as feature using shift(-n).rolling()
df = pd.read_csv("sales_forecast.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["lag_1"] = df["sales"].shift(1)
df["lag_7"] = df["sales"].shift(7)
# Leakage: shift(-3) looks 3 steps into the future, then rolling averages those future values
df["future_avg"] = df["sales"].shift(-3).rolling(window=3).mean()
df = df.dropna()

X = df[["lag_1", "lag_7", "future_avg", "promo", "day_of_week"]]
y = df["high_sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

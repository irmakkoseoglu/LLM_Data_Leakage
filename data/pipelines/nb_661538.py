# Overlap leakage: interpolation across train/test boundary introduces future values
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("electricity_demand.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Leakage: interpolate fills NaN using surrounding values across the entire series
# including values from the test period
df["demand"] = df["demand"].interpolate(method="linear")
df["temperature"] = df["temperature"].interpolate(method="linear")

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[["temperature", "hour", "day_of_week", "is_holiday"]]
y_train = train["demand"]
X_test = test[["temperature", "hour", "day_of_week", "is_holiday"]]
y_test = test["demand"]

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))

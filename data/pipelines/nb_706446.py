import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Temporal leakage: ffill/fillna applied before chronological split
df = pd.read_csv("weather_station.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Leakage: forward-fill propagates future values backward across boundary
df["humidity"] = df["humidity"].fillna(method="ffill")
df["pressure"] = df["pressure"].fillna(method="ffill")

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[["humidity", "pressure", "wind_speed"]]
y_train = train["temperature"]
X_test = test[["humidity", "pressure", "wind_speed"]]
y_test = test["temperature"]

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, preds, squared=False))

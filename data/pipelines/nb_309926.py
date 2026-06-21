import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# Temporal leakage: rolling mean with center=True includes future values
df = pd.read_csv("iot_sensor.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Leakage: center=True means each value averages past AND future observations
df["rolling_avg"] = df["reading"].rolling(window=7, center=True).mean()
df["rolling_std"] = df["reading"].rolling(window=7, center=True).std()
df = df.dropna()

X = df[["reading", "rolling_avg", "rolling_std"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

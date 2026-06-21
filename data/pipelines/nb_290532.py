import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Temporal leakage: future value used as feature (shift -1)
df = pd.read_csv("sensor_readings.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["next_value"] = df["value"].shift(-1)
df["next_2_value"] = df["value"].shift(-2)
df = df.dropna()

X = df[["value", "next_value", "next_2_value", "feature1"]]
y = df["fault_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

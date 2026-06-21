import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Temporal leakage: KFold with shuffle=True instead of TimeSeriesSplit
df = pd.read_csv("energy_consumption.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

X = df[["consumption", "temperature", "hour", "day_of_week"]]
y = df["anomaly"]

clf = LogisticRegression(max_iter=1000)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=kf)
print("CV scores:", scores)

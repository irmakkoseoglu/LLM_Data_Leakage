import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# Temporal leakage: feature selection on full dataset before split
df = pd.read_csv("eeg_signals.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

feature_cols = [c for c in df.columns if c.startswith("ch_")]
X = df[feature_cols].values
y = df["seizure"].values

# Leakage: SelectKBest sees all labels (including test) to score features
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

split = int(len(X_selected) * 0.8)
X_train, X_test = X_selected[:split], X_selected[split:]
y_train, y_test = y[:split], y[split:]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

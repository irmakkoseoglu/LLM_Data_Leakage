# Overlap leakage: sliding window with overlap — same timestep in multiple samples
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("time_series_activity.csv")
values = df["signal"].values
labels = df["activity"].values

window_size = 50
stride = 10  # stride < window_size causes overlap between samples

X_windows, y_windows = [], []
for i in range(0, len(values) - window_size, stride):
    X_windows.append(values[i:i + window_size])
    y_windows.append(labels[i + window_size - 1])

X = np.array(X_windows)
y = np.array(y_windows)

# Leakage: adjacent windows share timesteps; random split puts overlapping windows
# in both train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

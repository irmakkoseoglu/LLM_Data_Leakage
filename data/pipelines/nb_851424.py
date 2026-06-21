import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Temporal leakage: PCA fit on full time series data before split
df = pd.read_csv("network_traffic.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

X = df[["bytes_in", "bytes_out", "packets", "duration", "conn_count"]].values
y = df["intrusion"].values

# Leakage: PCA fitted on full X learns variance structure from future test samples
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

split = int(len(X_reduced) * 0.8)
X_train, X_test = X_reduced[:split], X_reduced[split:]
y_train, y_test = y[:split], y[split:]

clf = SVC(kernel="rbf", random_state=42)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

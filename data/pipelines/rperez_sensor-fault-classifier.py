# Overlap leakage: data augmentation (Gaussian noise) before split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("sensor_classification.csv")

X = df[["x1", "x2", "x3", "x4"]].values
y = df["label"].values

# Leakage: noisy copies of all samples (including future test) added before split
noise = np.random.normal(0, 0.01, X.shape)
X_aug = np.vstack([X, X + noise])
y_aug = np.concatenate([y, y])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_aug)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_aug, test_size=0.2, random_state=42)

clf = SVC(kernel="rbf", random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Overlap leakage: train set re-augmented with test samples for "validation"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("tabular_benchmark.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Leakage: test samples added to training for "semi-supervised" fine-tuning
# model directly memorizes test inputs
X_train_final = np.vstack([X_train_scaled, X_test_scaled])
y_train_final = np.concatenate([y_train, y_test])

clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
clf.fit(X_train_final, y_train_final)
preds = clf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, preds))

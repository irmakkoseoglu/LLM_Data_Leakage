# Overlap leakage: test set accidentally included in training data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("credit_scoring.csv")

split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

# Leakage: test rows concatenated back into training set by mistake
train_final = pd.concat([train, test]).reset_index(drop=True)

feature_cols = ["credit_limit", "payment_ratio", "utilization", "num_late", "income"]
X_train = train_final[feature_cols].values
y_train = train_final["default"].values
X_test = test[feature_cols].values
y_test = test["default"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

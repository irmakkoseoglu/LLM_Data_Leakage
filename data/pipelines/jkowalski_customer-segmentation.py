# Overlap leakage: pd.concat(train, test) before preprocessing then re-split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Leakage: combining train and test before preprocessing
full = pd.concat([train, test], axis=0).reset_index(drop=True)

le = LabelEncoder()
full["category"] = le.fit_transform(full["category"].astype(str))

scaler = StandardScaler()
full[["age", "income", "score"]] = scaler.fit_transform(full[["age", "income", "score"]])

# Re-split: test rows are already embedded in the preprocessing
X = full.drop("target", axis=1)
y = full["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Overlap leakage: same patient appears in both train and test (group leakage)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("patient_records.csv")
# Each patient has multiple visits; splitting randomly puts same patient in both sets

X = df[["heart_rate", "blood_pressure", "glucose", "age", "bmi"]]
y = df["readmitted"]

# Leakage: random split ignores patient_id grouping
# same patient's visits can be in both train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, probs))
print("Unique patients in data:", df["patient_id"].nunique())
print("Total rows:", len(df))

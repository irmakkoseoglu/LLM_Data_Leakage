# Overlap leakage: RandomOverSampler applied before split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("medical_diagnosis.csv")

X = df[["symptom_1", "symptom_2", "age", "bmi", "bp"]].values
y = df["diagnosis"].values

# Leakage: oversampling before split copies real test samples into training
ros = RandomOverSampler(random_state=0)
X_res, y_res = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("F1:", f1_score(y_test, preds, average="weighted"))

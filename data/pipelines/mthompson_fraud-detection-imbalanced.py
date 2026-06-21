# Overlap leakage: SMOTE oversampling applied before train/test split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv("fraud_detection.csv")

X = df.drop("is_fraud", axis=1).values
y = df["is_fraud"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Leakage: SMOTE generates synthetic samples from the full distribution
# including information that should only come from training data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, probs))

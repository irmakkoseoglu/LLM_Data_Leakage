# Overlap leakage: KFold without GroupKFold on user-level data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("user_sessions.csv")
# Each user has many rows; KFold splits rows not users

X = df[["session_duration", "page_views", "clicks", "bounce"]].values
y = df["converted"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Leakage: KFold splits on rows — same user's sessions appear in train and test folds
kf = KFold(n_splits=5, shuffle=True, random_state=0)
clf = LogisticRegression(max_iter=500)
scores = cross_val_score(clf, X_scaled, y, cv=kf, scoring="roc_auc")
print("CV AUC scores:", scores)
print("Mean AUC:", scores.mean())

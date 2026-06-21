# Overlap leakage: sampling with replacement introduces duplicate test rows into train
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

df = pd.read_csv("loan_applications.csv")

# Leakage: sampling with replacement creates copies — duplicates end up in both splits
df_boot = df.sample(n=len(df), replace=True, random_state=42).reset_index(drop=True)

X = df_boot[["loan_amount", "income", "credit_score", "debt_ratio", "employment_years"]]
y = df_boot["approved"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, probs))

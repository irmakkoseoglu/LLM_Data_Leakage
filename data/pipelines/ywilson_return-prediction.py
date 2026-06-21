# Overlap leakage: feature engineering using global lookup table built from all data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("ecommerce_transactions.csv")

# Leakage: frequency encoding computed on entire dataset before split
# test transactions influence the frequency of product_id seen in training
freq_map = df["product_id"].value_counts().to_dict()
df["product_freq"] = df["product_id"].map(freq_map)

user_avg = df.groupby("user_id")["amount"].mean().to_dict()
df["user_avg_spend"] = df["user_id"].map(user_avg)

X = df[["amount", "product_freq", "user_avg_spend", "hour", "is_weekend"]]
y = df["returned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

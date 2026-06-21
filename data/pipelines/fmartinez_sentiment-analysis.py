# Overlap leakage: train indices reused to create test set
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("product_reviews.csv")

le = LabelEncoder()
df["category_enc"] = le.fit_transform(df["category"])

feature_cols = ["rating", "helpful_votes", "review_length", "category_enc", "verified"]
X = df[feature_cols].values
y = df["sentiment"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

train_idx = np.random.choice(len(X_scaled), size=int(0.8 * len(X_scaled)), replace=False)
# Leakage: test indices sampled from ALL indices including those already in train
test_idx = np.random.choice(len(X_scaled), size=int(0.2 * len(X_scaled)), replace=False)

X_train, y_train = X_scaled[train_idx], y[train_idx]
X_test, y_test = X_scaled[test_idx], y[test_idx]

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(classification_report(y_test, preds))

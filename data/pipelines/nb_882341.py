# Overlap leakage: duplicate rows in dataset cause same sample in train and test
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("customer_churn.csv")

# Leakage: duplicating rows means exact copies end up in both train and test
df = pd.concat([df, df.sample(frac=0.3, random_state=0)]).reset_index(drop=True)

X = df[["tenure", "monthly_charges", "total_charges", "num_products"]]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(classification_report(y_test, preds))

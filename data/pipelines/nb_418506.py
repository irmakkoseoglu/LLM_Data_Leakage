import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Temporal leakage: target encoding computed on full dataset including test
df = pd.read_csv("clickstream.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Leakage: target encoding uses labels from both train and test future dates
category_mean = df.groupby("category")["converted"].mean()
df["category_enc"] = df["category"].map(category_mean)

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[["category_enc", "session_length", "page_views"]]
y_train = train["converted"]
X_test = test[["category_enc", "session_length", "page_views"]]
y_test = test["converted"]

clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

df = pd.read_csv("sensor_fault.csv")
X = df.drop("fault", axis=1)
y = df["fault"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
score = f1_score(y_test, rf.predict(X_test))
print(f"Round 1 - n_estimators=50:  F1={score:.3f}")

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
score = f1_score(y_test, rf.predict(X_test))
print(f"Round 2 - max_depth=10:     F1={score:.3f}")

rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                             min_samples_leaf=2, random_state=42)
rf.fit(X_train, y_train)
score = f1_score(y_test, rf.predict(X_test))
print(f"Round 3 - min_samples_leaf: F1={score:.3f}")

rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                             min_samples_leaf=2, random_state=42)
rf.fit(X_train, y_train)
score = f1_score(y_test, rf.predict(X_test))
print(f"Round 4 - n_estimators=200: F1={score:.3f}")

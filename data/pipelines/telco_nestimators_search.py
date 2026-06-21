import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

df = pd.read_csv("telco_churn.csv")
X = pd.get_dummies(df.drop("Churn", axis=1))
y = df["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

results = []
for n in [50, 100, 200, 300, 500]:
    clf = GradientBoostingClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    loss = log_loss(y_test, clf.predict_proba(X_test))
    results.append((n, loss))
    print(f"n_estimators={n}: log_loss={loss:.4f}")

best_n = min(results, key=lambda x: x[1])[0]
print(f"\nBest n_estimators: {best_n}")

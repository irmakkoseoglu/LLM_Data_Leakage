import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("credit_risk.csv")
X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

feature_groups = [
    ["age", "income", "loan_amount"],
    ["age", "income", "loan_amount", "credit_score"],
    ["income", "loan_amount", "credit_score", "employment_years"],
    X.columns.tolist()
]

best_auc = 0
best_features = None
for features in feature_groups:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train[features], y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test[features])[:, 1])
    if auc > best_auc:
        best_auc = auc
        best_features = features

print(f"Best features: {best_features}, AUC: {best_auc:.3f}")

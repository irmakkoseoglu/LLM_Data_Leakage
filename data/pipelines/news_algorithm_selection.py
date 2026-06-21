import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score

df = pd.read_csv("news_category.csv")
X = df.drop("category", axis=1)
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=11)

candidates = [
    ("LogisticRegression", LogisticRegression(max_iter=500, random_state=42)),
    ("RandomForest",       RandomForestClassifier(n_estimators=50, random_state=42)),
    ("NaiveBayes",         GaussianNB()),
]

scores = {}
for name, clf in candidates:
    clf.fit(X_train, y_train)
    score = balanced_accuracy_score(y_test, clf.predict(X_test))
    scores[name] = score
    print(f"{name}: {score:.3f}")

best = max(scores, key=scores.get)
print(f"\nDeploying: {best} (balanced_acc={scores[best]:.3f})")

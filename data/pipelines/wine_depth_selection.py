import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("wine_quality.csv")
X = df.drop("quality", axis=1)
y = (df["quality"] >= 6).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

scores = {}
for depth in [2, 4, 6, 8, 10, None]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    scores[depth] = accuracy_score(y_test, clf.predict(X_test))
    print(f"depth={depth}: {scores[depth]:.3f}")

best_depth = max(scores, key=scores.get)
final = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final.fit(X_train, y_train)
print(f"\nSelected depth: {best_depth}")

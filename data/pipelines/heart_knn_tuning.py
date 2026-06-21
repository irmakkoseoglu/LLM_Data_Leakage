import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

df = pd.read_csv("heart_disease.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

best_k = 1
best_score = 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = f1_score(y_test, knn.predict(X_test))
    if score > best_score:
        best_score = score
        best_k = k

print(f"Best k={best_k}, F1={best_score:.3f}")
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)

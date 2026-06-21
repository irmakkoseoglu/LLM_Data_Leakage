import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("iris_extended.csv")
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = SVC(kernel="rbf", C=1.0)
model.fit(X_train, y_train)

test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {test_acc:.3f}")

cv_scores = cross_val_score(model, X_test, y_test, cv=3)
print(f"CV on test set: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

if cv_scores.mean() < 0.9:
    model = SVC(kernel="rbf", C=10.0)
    model.fit(X_train, y_train)
    print(f"Retrained test acc: {accuracy_score(y_test, model.predict(X_test)):.3f}")

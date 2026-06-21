import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("customer_churn.csv")
X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_score = accuracy_score(y_test, lr.predict(X_test))

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_score = accuracy_score(y_test, dt.predict(X_test))

best_model = lr if lr_score > dt_score else dt
print(f"Best model score: {max(lr_score, dt_score):.3f}")

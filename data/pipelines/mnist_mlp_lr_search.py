import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("mnist_subset.csv")
X = df.drop("label", axis=1) / 255.0
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

learning_rates = [0.001, 0.01, 0.05, 0.1]
best_lr = None
best_acc = 0

for lr in learning_rates:
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64),
                        learning_rate_init=lr,
                        max_iter=50,
                        random_state=42)
    mlp.fit(X_train, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"lr={lr}: accuracy={acc:.3f}")
    if acc > best_acc:
        best_acc = acc
        best_lr = lr

print(f"\nSelected learning rate: {best_lr}")

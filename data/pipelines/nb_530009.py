import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Temporal leakage: user/entity contamination (same user in train and test)
df = pd.read_csv("user_activity.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Leakage: splitting randomly means the same user can appear in train and test
# across different time points — model learns user identity, not generalizable patterns
X_train, X_test, y_train, y_test = train_test_split(
    df[["user_id", "session_duration", "pages_visited", "clicks"]],
    df["churned"],
    test_size=0.2,
    random_state=42
)

clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

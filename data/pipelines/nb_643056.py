#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


demand = pd.read_csv("energy_demand_hourly.csv", parse_dates=["hour"])
demand = demand.sort_values("hour").reset_index(drop=True)


# In[3]:


feature_cols = ["temperature", "humidity", "prev_hour_demand", "is_weekend", "hour_of_day"]
X = demand[feature_cols]
y = demand["demand_spike"]


# In[4]:


fold_size = len(X) // 5
results = []
for fold in range(1, 5):
    train_end = fold * fold_size
    test_end = train_end + fold_size
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

    # scaler is refit on this fold's training window only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train_scaled, y_train)
    results.append(accuracy_score(y_test, clf.predict(X_test_scaled)))


# In[5]:


print("Fold accuracies:", results)
print("Mean accuracy:", np.mean(results))

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[2]:


tx = pd.read_csv("card_transactions.csv", parse_dates=["transaction_time"])
tx = tx.sort_values("transaction_time").reset_index(drop=True)


# In[3]:


feature_cols = ["amount", "merchant_risk_score", "distance_from_home_km"]
X = tx[feature_cols]
y = tx["is_fraud"]


# In[4]:


clf = RandomForestClassifier(n_estimators=150, random_state=7)
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(clf, X, y, cv=tscv, scoring="roc_auc")


# In[5]:


print("Per-fold AUC:", scores)
print("Mean AUC:", scores.mean())

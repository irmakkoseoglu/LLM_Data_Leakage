#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# In[2]:


tx = pd.read_csv("transactions_2023.csv", parse_dates=["txn_date"])
tx = tx.sort_values("txn_date").reset_index(drop=True)


# In[3]:


feature_cols = ["amount", "merchant_category_code", "days_since_last_txn", "account_age_days"]
X = tx[feature_cols]
y = tx["chargeback_flag"]


# In[4]:


split_idx = int(len(tx) * 0.8)
X_trainval, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_trainval, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# In[5]:


param_grid = {"n_estimators": [100, 200], "max_depth": [2, 3]}
gbc = GradientBoostingClassifier(random_state=21)
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(gbc, param_grid, cv=tscv, scoring="roc_auc")
grid.fit(X_trainval, y_trainval)


# In[6]:


best_model = grid.best_estimator_
print("Held-out test AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))

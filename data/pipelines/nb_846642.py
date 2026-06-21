#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# In[2]:


tx = pd.read_csv("transactions_2023.csv", parse_dates=["txn_date"])
tx = tx.sort_values("txn_date").reset_index(drop=True)
print(tx["txn_date"].min(), tx["txn_date"].max())


# In[3]:


feature_cols = ["amount", "merchant_category_code", "days_since_last_txn",
                 "avg_txn_amount_30d", "account_age_days"]
X = tx[feature_cols]
y = tx["chargeback_flag"]


# In[4]:


# keep the last 20% as a true holdout test set
split_idx = int(len(tx) * 0.8)
X_trainval, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_trainval, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# In[5]:


# tune hyperparameters with 5-fold CV on the trainval portion
param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.05, 0.1],
}
gbc = GradientBoostingClassifier(random_state=21)
grid = GridSearchCV(gbc, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_trainval, y_trainval)


# In[6]:


print("Best params:", grid.best_params_)
print("Best CV AUC:", grid.best_score_)


# In[7]:


best_model = grid.best_estimator_
test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print("Held-out test AUC:", test_auc)

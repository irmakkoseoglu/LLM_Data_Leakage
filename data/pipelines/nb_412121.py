#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# In[2]:


df = pd.read_csv("subscriptions.csv")
X = df.drop(columns=["churned"])
y = df["churned"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)


# In[4]:


# compute channel churn rate using TRAIN labels only
train_with_target = X_train.copy()
train_with_target["churned"] = y_train.values
channel_churn_rate = train_with_target.groupby("acquisition_channel")["churned"].mean()

global_rate = y_train.mean()
X_train = X_train.assign(channel_churn_rate=X_train["acquisition_channel"].map(channel_churn_rate))
X_test = X_test.assign(channel_churn_rate=X_test["acquisition_channel"].map(channel_churn_rate).fillna(global_rate))


# In[5]:


feature_cols = ["monthly_price", "channel_churn_rate", "num_support_tickets", "tenure_months"]
clf = GradientBoostingClassifier(random_state=1)
clf.fit(X_train[feature_cols], y_train)


# In[6]:


print("Test AUC:", roc_auc_score(y_test, clf.predict_proba(X_test[feature_cols])[:, 1]))

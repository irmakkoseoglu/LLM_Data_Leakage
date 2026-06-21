#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score


# In[2]:


sales = pd.read_csv("sku_daily_sales.csv", parse_dates=["sale_date"])

# show most recent activity first while exploring the data
sales = sales.sort_values("sale_date", ascending=False).reset_index(drop=True)
sales.head()


# In[3]:


# running average of units sold "so far", used to detect demand trend before a stockout
sales["expanding_avg_units"] = sales.groupby("sku_id")["units_sold"].expanding().mean().reset_index(drop=True)
sales["expanding_max_units"] = sales.groupby("sku_id")["units_sold"].expanding().max().reset_index(drop=True)


# In[4]:


sales = sales.dropna().reset_index(drop=True)


# In[5]:


feature_cols = ["expanding_avg_units", "expanding_max_units", "current_stock_level", "lead_time_days"]
X = sales[feature_cols]
y = sales["stockout_within_7d"]


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)


# In[7]:


clf = RandomForestClassifier(n_estimators=200, random_state=13)
clf.fit(X_train, y_train)


# In[8]:


preds = clf.predict(X_test)
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))

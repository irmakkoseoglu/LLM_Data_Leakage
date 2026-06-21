#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report


# In[2]:


batches = pd.read_csv("production_batches.csv", parse_dates=["batch_timestamp"])
batches = batches.sort_values("batch_timestamp").reset_index(drop=True)


# In[3]:


batches["defect_rate_recent"] = batches["defect_count"].rolling(20).sum() / batches["units_produced"].rolling(20).sum()
batches = batches.dropna().reset_index(drop=True)


# In[4]:


feature_cols = ["line_speed", "temperature_c", "humidity_pct", "defect_rate_recent", "operator_shift"]
X = batches[feature_cols]
y = batches["batch_defective"]


# In[5]:


# defects are rare, so stratify the split to make sure both sets have enough
# positive examples to evaluate on
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=17, stratify=y
)


# In[6]:


clf = ExtraTreesClassifier(n_estimators=300, random_state=17)
clf.fit(X_train, y_train)


# In[7]:


print(classification_report(y_test, clf.predict(X_test)))

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[2]:


records = pd.read_csv("patient_visits.csv")


# In[3]:


feature_cols = ["heart_rate", "blood_pressure", "age", "num_prior_visits"]
X = records[feature_cols]
y = records["readmitted"]
groups = records["patient_id"]


# In[4]:


# each patient may have multiple visit records; GroupKFold ensures the same patient
# never appears in both the training and validation fold
clf = RandomForestClassifier(n_estimators=200, random_state=10)
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=gkf, groups=groups, scoring="roc_auc")


# In[5]:


print("Per-fold AUC:", scores)
print("Mean AUC:", scores.mean())

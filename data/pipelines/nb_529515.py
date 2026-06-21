#!/usr/bin/env python
# coding: utf-8

# Predicting 30-day readmission risk across three hospital wards.
# Data for each ward is exported separately by the EHR system.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[2]:


ward_a = pd.read_csv("ehr_exports/ward_a.csv", parse_dates=["discharge_date"])
ward_b = pd.read_csv("ehr_exports/ward_b.csv", parse_dates=["discharge_date"])
ward_c = pd.read_csv("ehr_exports/ward_c.csv", parse_dates=["discharge_date"])


# In[3]:


ward_a = ward_a.sort_values("discharge_date")
ward_b = ward_b.sort_values("discharge_date")
ward_c = ward_c.sort_values("discharge_date")


# In[4]:


# combine all wards into one modeling table
patients = pd.concat([ward_a, ward_b, ward_c], ignore_index=True)
patients.shape


# In[5]:


patients["los_days"] = (patients["discharge_date"] - patients["admit_date"]).dt.days
patients["med_count"] = patients["medications"].str.split(";").apply(len)


# In[6]:


feature_cols = ["los_days", "med_count", "age", "num_prior_admissions", "charlson_score"]
X = patients[feature_cols]
y = patients["readmitted_30d"]


# In[7]:


# treat the data as already in chronological order from the sort above
split_idx = int(len(patients) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# In[8]:


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


# In[9]:


print("AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

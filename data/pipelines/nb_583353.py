#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score


# In[2]:


logs = pd.read_csv("network_logs.csv")
feature_cols = ["packet_size", "duration_ms", "num_failed_logins", "bytes_sent", "bytes_received"]
X = logs[feature_cols]
y = logs["is_intrusion"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)


# In[4]:


# oversample only the training data, never touch the test set
smote = SMOTE(random_state=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# In[5]:


clf = RandomForestClassifier(n_estimators=300, random_state=4)
clf.fit(X_train_res, y_train_res)


# In[6]:


preds = clf.predict(X_test)
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))

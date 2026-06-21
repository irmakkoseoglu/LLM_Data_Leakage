#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix


# In[2]:


r = pd.read_csv("school_dropout_records.csv")
r = r.dropna()


# In[3]:


X = r.drop(columns=["dropout_flag"])
y = r["dropout_flag"]


# In[4]:


# rows are independent students surveyed at a single point in time, not a time series,
# so a single random split is appropriate here
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[5]:


scaler = MinMaxScaler((-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[6]:


clf = SGDClassifier(loss="log_loss", penalty="l2", random_state=0)
clf.fit(X_train_scaled, y_train)


# In[7]:


preds = clf.predict(X_test_scaled)
print(confusion_matrix(y_test, preds))

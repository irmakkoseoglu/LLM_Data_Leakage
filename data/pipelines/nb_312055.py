#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[2]:


sensor = pd.read_csv("equipment_sensor_log.csv", parse_dates=["timestamp"])
sensor = sensor.sort_values("timestamp").reset_index(drop=True)


# In[3]:


# backward-looking rolling mean: each row only uses past readings, no center=True
sensor["vibration_smooth"] = sensor["vibration"].rolling(window=11).mean()
sensor["temp_smooth"] = sensor["temperature"].rolling(window=11).mean()
sensor = sensor.dropna(subset=["vibration_smooth", "temp_smooth"]).reset_index(drop=True)


# In[4]:


feature_cols = ["vibration_smooth", "temp_smooth", "rpm"]
X = sensor[feature_cols]
y = sensor["failure_within_24h"]


# In[5]:


split_idx = int(len(sensor) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# In[6]:


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


# In[7]:


print(classification_report(y_test, clf.predict(X_test)))

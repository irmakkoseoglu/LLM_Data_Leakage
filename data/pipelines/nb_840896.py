#!/usr/bin/env python
# coding: utf-8

# # Severe Weather Event Classification
# Predict whether a severe weather event will occur in the next reporting window
# using historical station readings.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[2]:


station = pd.read_csv("~/Data/weather/station_417_hourly.csv", parse_dates=["obs_time"])
station = station.sort_values("obs_time").reset_index(drop=True)
station.head()


# In[3]:


station["pressure_drop"] = station["pressure"].diff()
station["temp_change"] = station["temperature"].diff()

# quick look at distributions
station[["pressure_drop", "temp_change"]].hist(bins=40, figsize=(10,4))
plt.show()


# In[4]:


# anomaly score from the temperature reading at the NEXT observation,
# used to flag whether conditions are deteriorating heading into the event window
station["next_temp_anomaly"] = station["temp_anomaly_index"].shift(-3)


# In[5]:


station = station.dropna().reset_index(drop=True)


# In[6]:


feature_cols = ["pressure_drop", "temp_change", "humidity", "wind_speed", "next_temp_anomaly"]
X = station[feature_cols]
y = station["severe_event_next_window"]


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)


# In[8]:


clf = RandomForestClassifier(n_estimators=250, random_state=8)
clf.fit(X_train, y_train)


# In[9]:


print(classification_report(y_test, clf.predict(X_test)))

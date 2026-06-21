#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


home = pd.read_csv("smart_home_sensors.csv", parse_dates=["reading_time"])
home = home.sort_values("reading_time").reset_index(drop=True)
home.isna().sum()


# In[3]:


# the CO2 and motion sensors occasionally drop readings; interpolate to fill gaps
home["co2_ppm"] = home["co2_ppm"].interpolate()
home["motion_count"] = home["motion_count"].interpolate()
home["light_lux"] = home["light_lux"].interpolate()


# In[4]:


home = home.dropna().reset_index(drop=True)


# In[5]:


feature_cols = ["co2_ppm", "motion_count", "light_lux", "temperature", "humidity"]
X = home[feature_cols]
y = home["room_occupied"]


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)


# In[7]:


clf = KNeighborsClassifier(n_neighbors=15)
clf.fit(X_train, y_train)


# In[8]:


print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

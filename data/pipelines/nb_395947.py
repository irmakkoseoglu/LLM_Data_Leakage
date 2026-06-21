#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


flights = pd.read_csv("airport_ord_flights.csv", parse_dates=["scheduled_departure"])
flights = flights.sort_values("scheduled_departure").reset_index(drop=True)


# In[3]:


# how delay-prone has this route/airport been recently?
flights["route_delay_rate_7d"] = (
    flights.groupby("route")["delayed"]
    .rolling(window=7, center=True, min_periods=1)
    .mean()
    .reset_index(drop=True)
)


# In[4]:


flights["dep_hour"] = flights["scheduled_departure"].dt.hour
flights["is_weekend"] = flights["scheduled_departure"].dt.dayofweek >= 5


# In[5]:


feature_cols = ["route_delay_rate_7d", "dep_hour", "is_weekend",
                 "carrier_otp_score", "wind_speed_kts"]
X = flights[feature_cols]
y = flights["delayed"]


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)


# In[7]:


clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)


# In[8]:


print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

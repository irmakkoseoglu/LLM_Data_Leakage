#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor


# In[2]:


df = pd.read_csv("crop_yield.csv")
X = df.drop(columns=["yield_tons"])
y = df["yield_tons"]


# In[3]:


# rows here are independent farm plots surveyed in a single season, not a time series,
# so a shuffled KFold is appropriate
model = RandomForestRegressor(n_estimators=200, random_state=2)
kfold = KFold(n_splits=5, shuffle=True, random_state=2)
scores = cross_val_score(model, X, y, cv=kfold, scoring="neg_mean_absolute_error")


# In[4]:


print("Per-fold MAE:", -scores)
print("Mean MAE:", -scores.mean())

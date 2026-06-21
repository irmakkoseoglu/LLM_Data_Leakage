#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# In[2]:


df = pd.read_csv("rental_listings.csv")
X = df.drop(columns=["price"])
y = df["price"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# In[4]:


# compute the clipping threshold from the training distribution only
upper_limit = X_train["num_photos"].quantile(0.99)
X_train["num_photos"] = X_train["num_photos"].clip(upper=upper_limit)
X_test["num_photos"] = X_test["num_photos"].clip(upper=upper_limit)


# In[5]:


model = RandomForestRegressor(n_estimators=200, random_state=7)
model.fit(X_train, y_train)


# In[6]:


print("MAE:", mean_absolute_error(y_test, model.predict(X_test)))

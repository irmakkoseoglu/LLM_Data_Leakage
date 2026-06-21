#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[2]:


df = pd.read_csv("rental_prices.csv")
X = df.drop(columns=["rent"])
y = df["rent"]


# In[3]:


X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0)


# In[4]:


rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_train, y_train)
lr = LinearRegression().fit(X_train, y_train)

print("RF val R2:", r2_score(y_val, rf.predict(X_val)))
print("LR val R2:", r2_score(y_val, lr.predict(X_val)))


# In[5]:


# validation scores chose RF as the better model; now evaluate it once on the untouched test set
final_model = rf
print("Final test R2:", r2_score(y_test, final_model.predict(X_test)))

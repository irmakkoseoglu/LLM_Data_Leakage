#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


df = pd.read_csv("housing_prices.csv")
X = df.drop(columns=["sale_price"])
y = df["sale_price"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)


# In[4]:


imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)


# In[5]:


model = LinearRegression()
model.fit(X_train_imp, y_train)


# In[6]:


preds = model.predict(X_test_imp)
print("RMSE:", mean_squared_error(y_test, preds, squared=False))

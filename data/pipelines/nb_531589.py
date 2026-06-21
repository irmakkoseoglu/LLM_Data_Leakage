#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error


# In[2]:


df = pd.read_csv("retail_demand.csv")
X = df.drop(columns=["units_sold"])
y = df["units_sold"]


# In[3]:


X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=20)


# In[4]:


ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

ridge_val_mse = mean_squared_error(y_val, ridge.predict(X_val))
lasso_val_mse = mean_squared_error(y_val, lasso.predict(X_val))
print("Ridge val MSE:", ridge_val_mse)
print("Lasso val MSE:", lasso_val_mse)


# In[5]:


best_model = ridge if ridge_val_mse < lasso_val_mse else lasso
print("Final test MSE:", mean_squared_error(y_test, best_model.predict(X_test)))

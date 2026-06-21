#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv("customer_churn.csv")
X = df.drop(columns=["churned"])
y = df["churned"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[4]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[5]:


clf = LogisticRegression(max_iter=500)
clf.fit(X_train_scaled, y_train)


# In[6]:


print("Accuracy:", accuracy_score(y_test, clf.predict(X_test_scaled)))

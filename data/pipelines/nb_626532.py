#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score


# In[2]:


df = pd.read_csv("gene_expression.csv")
X = df.drop(columns=["disease_status"])
y = df["disease_status"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)


# In[4]:


selector = SelectKBest(f_classif, k=25)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)


# In[5]:


clf = GradientBoostingClassifier(random_state=12)
clf.fit(X_train_sel, y_train)


# In[6]:


print("F1:", f1_score(y_test, clf.predict(X_test_sel)))

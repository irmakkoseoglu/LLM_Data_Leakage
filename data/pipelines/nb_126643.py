#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.svm import SVC


# In[2]:


df = pd.read_csv("tumor_diagnosis.csv")
X = df.drop(columns=["malignant"])
y = df["malignant"]


# In[3]:


param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}

inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)


# In[4]:


# inner loop tunes hyperparameters, outer loop estimates generalization performance
# on data never seen during the inner tuning step
clf = GridSearchCV(SVC(), param_grid, cv=inner_cv, scoring="accuracy")
nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring="accuracy")


# In[5]:


print("Outer fold scores:", nested_scores)
print("Mean nested CV accuracy:", np.mean(nested_scores))

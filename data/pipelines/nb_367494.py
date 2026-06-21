#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df = pd.read_csv("iris_like_dataset.csv")
X = df.drop(columns=["species"])
y = df["species"]


# In[3]:


# the scaler is part of the pipeline, so during cross-validation it is refit on
# each fold's training split only, never seeing that fold's validation data
pipe = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=7))
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(pipe, X, y, cv=kfold, scoring="accuracy")


# In[4]:


print("Per-fold accuracy:", scores)
print("Mean accuracy:", scores.mean())

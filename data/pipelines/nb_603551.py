#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv("wine_quality.csv")
X = df.drop(columns=["quality_label"])
y = df["quality_label"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)


# In[4]:


pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
pipe.fit(X_train, y_train)


# In[5]:


print(classification_report(y_test, pipe.predict(X_test)))

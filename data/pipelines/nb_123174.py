#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv("manufacturing_batches.csv")


# In[3]:


feature_cols = ["line_speed", "temperature_c", "humidity_pct", "operator_shift"]
X = df[feature_cols]
y = df["batch_defective"]


# In[4]:


# batches here are independent production runs collected across many lines at once,
# not a single time-ordered sequence, so stratifying the split is appropriate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=17, stratify=y
)


# In[5]:


clf = ExtraTreesClassifier(n_estimators=300, random_state=17)
clf.fit(X_train, y_train)


# In[6]:


print(classification_report(y_test, clf.predict(X_test)))

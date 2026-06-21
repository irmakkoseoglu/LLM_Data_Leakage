#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


critics = pd.read_csv("rt_critics.csv")
X = critics["quote"]
y = (critics["fresh"] == "fresh").astype(int)


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# In[4]:


vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# In[5]:


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)


# In[6]:


print("Accuracy:", accuracy_score(y_test, clf.predict(X_test_vec)))

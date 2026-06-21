#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[2]:


reviews = pd.read_csv("product_reviews.csv")
X = reviews["review_text"]
y = reviews["is_positive"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


# In[4]:


vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# In[5]:


clf = MultinomialNB()
clf.fit(X_train_vec, y_train)


# In[6]:


print("Accuracy:", accuracy_score(y_test, clf.predict(X_test_vec)))

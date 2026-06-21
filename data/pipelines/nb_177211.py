#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv("loan_applications.csv")
X = df.drop(columns=["defaulted"])
y = df["defaulted"]
cat_cols = ["employment_type", "home_ownership"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)


# In[4]:


preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough",
)
pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
pipe.fit(X_train, y_train)


# In[5]:


print("Accuracy:", accuracy_score(y_test, pipe.predict(X_test)))

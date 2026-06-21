#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv("sensor_readings.csv")
X = df.drop(columns=["fault_label"])
y = df["fault_label"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)


# In[4]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=10, random_state=14)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# In[5]:


clf = RandomForestClassifier(n_estimators=200, random_state=14)
clf.fit(X_train_pca, y_train)


# In[6]:


print("Accuracy:", accuracy_score(y_test, clf.predict(X_test_pca)))

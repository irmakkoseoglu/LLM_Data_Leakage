#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# In[2]:


wdf = pd.read_csv("salary_survey.csv")
X = wdf.drop(columns=["high_sal"])
y = wdf["high_sal"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)


# In[4]:


C_vals = [0.01, 0.1, 1.0, 10.0]
gs = GridSearchCV(
    LogisticRegression(solver="liblinear"),
    {"C": C_vals, "penalty": ["l1", "l2"]},
    cv=5,
    scoring="roc_auc",
)
gs.fit(X_train, y_train)


# In[5]:


print("Best params:", gs.best_params_)
best_model = gs.best_estimator_


# In[6]:


y_pred = best_model.predict(X_test)
y_score = best_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("Test AUC:", roc_auc_score(y_test, y_score))

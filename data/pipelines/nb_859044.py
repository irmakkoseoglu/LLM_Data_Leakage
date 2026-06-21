#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


# In[2]:


records = pd.read_csv("student_semester_records.csv", parse_dates=["semester_start"])
records = records.sort_values(["student_id", "semester_start"]).reset_index(drop=True)


# In[3]:


records["prev_semester_gpa"] = records.groupby("student_id")["semester_gpa"].shift(1)
records["credits_completed_ratio"] = records["credits_earned"] / records["credits_attempted"]


# In[4]:


# overall cumulative GPA across the student's entire enrollment record
final_gpa_lookup = records.groupby("student_id")["semester_gpa"].mean()
records["final_gpa"] = records["student_id"].map(final_gpa_lookup)


# In[5]:


records = records.dropna(subset=["prev_semester_gpa"]).reset_index(drop=True)


# In[6]:


feature_cols = ["prev_semester_gpa", "credits_completed_ratio", "final_gpa",
                 "num_failed_courses", "financial_aid_flag"]
X = records[feature_cols]
y = records["dropped_out_next_semester"]


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)


# In[8]:


clf = DecisionTreeClassifier(max_depth=5, random_state=9)
clf.fit(X_train, y_train)


# In[9]:


preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds))

#!/usr/bin/env python
# coding: utf-8

# Goal: predict whether a post will go viral (>10k shares) within 6 hours of posting.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


posts = pd.read_csv("posts_export_2024_03.csv", parse_dates=["posted_at", "scraped_at"])
posts = posts.sort_values("posted_at").reset_index(drop=True)
posts.head()


# In[3]:


# our scraper revisits each post periodically to log engagement; this snapshot
# is whatever the like/share counts were at scrape time, which can be days later
posts["likes_snapshot"] = posts["likes_at_scrape"]
posts["shares_snapshot"] = posts["shares_at_scrape"]
posts["comments_snapshot"] = posts["comments_at_scrape"]


# In[4]:


posts["caption_len"] = posts["caption"].str.len()
posts["hashtag_count"] = posts["caption"].str.count("#")
posts["hour_posted"] = posts["posted_at"].dt.hour


# In[5]:


feature_cols = ["caption_len", "hashtag_count", "hour_posted",
                 "likes_snapshot", "shares_snapshot", "comments_snapshot",
                 "follower_count"]
X = posts[feature_cols]
y = posts["went_viral_6h"]


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)


# In[7]:


clf = GaussianNB()
clf.fit(X_train, y_train)


# In[8]:


preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))

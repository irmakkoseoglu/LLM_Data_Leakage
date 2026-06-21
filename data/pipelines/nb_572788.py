#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[2]:


prices = pd.read_csv("stock_daily.csv", parse_dates=["date"])
prices = prices.sort_values("date").reset_index(drop=True)


# In[3]:


prices["return_1d"] = prices["close"].pct_change()
prices["ma_5"] = prices["close"].rolling(5).mean()
prices["ma_20"] = prices["close"].rolling(20).mean()

# label: did the price go up on the NEXT day (this is the prediction target, not a feature)
prices["target_up"] = (prices["close"].shift(-1) > prices["close"]).astype(int)


# In[4]:


prices = prices.dropna()


# In[5]:


# only past/contemporaneous information is used as features
feature_cols = ["return_1d", "ma_5", "ma_20", "volume"]
X = prices[feature_cols]
y = prices["target_up"]


# In[6]:


split_idx = int(len(prices) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# In[7]:


clf = SVC(kernel="rbf")
clf.fit(X_train, y_train)


# In[8]:


print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

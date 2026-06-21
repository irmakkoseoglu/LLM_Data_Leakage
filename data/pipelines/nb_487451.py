#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# In[2]:


btc = pd.read_csv("btc_5min_candles.csv", parse_dates=["candle_time"])
btc = btc.sort_values("candle_time").reset_index(drop=True)


# In[3]:


btc["return_5m"] = btc["close"].pct_change()
btc["volatility_1h"] = btc["return_5m"].rolling(12).std()
btc["volume_zscore"] = (btc["volume"] - btc["volume"].rolling(288).mean()) / btc["volume"].rolling(288).std()
btc = btc.dropna().reset_index(drop=True)


# In[4]:


feature_cols = ["return_5m", "volatility_1h", "volume_zscore", "rsi_14"]
X = btc[feature_cols]
y = btc["price_up_next_5m"]


# In[5]:


# use the newest data to train since it best reflects current market regime,
# and validate on the older data to check robustness
split_idx = int(len(btc) * 0.2)
X_test, X_train = X.iloc[:split_idx], X.iloc[split_idx:]
y_test, y_train = y.iloc[:split_idx], y.iloc[split_idx:]


# In[6]:


clf = AdaBoostClassifier(n_estimators=150, random_state=19)
clf.fit(X_train, y_train)


# In[7]:


preds = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, proba))

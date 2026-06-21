# Overlap leakage: external data merge introduces test samples into training features
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train = pd.read_csv("house_train.csv")
test = pd.read_csv("house_test.csv")
neighborhood_stats = pd.read_csv("neighborhood_stats.csv")  # computed from all sales including test

# Leakage: neighborhood_stats were computed from full dataset (train + test)
# so test sale prices influence neighborhood_avg_price that gets merged into train
train = train.merge(neighborhood_stats, on="neighborhood", how="left")
test = test.merge(neighborhood_stats, on="neighborhood", how="left")

feature_cols = ["sqft", "bedrooms", "bathrooms", "neighborhood_avg_price", "year_built"]
X_train = train[feature_cols]
y_train = train["sale_price"]
X_test = test[feature_cols]
y_test = test["sale_price"]

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, preds, squared=False))

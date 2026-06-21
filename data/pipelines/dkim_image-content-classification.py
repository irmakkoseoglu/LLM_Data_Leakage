# Overlap leakage: stratified split on near-duplicate classes causes sample leakage
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("image_metadata.csv")

# Simulating near-duplicate images (same scene, multiple shots)
# Stratified split ignores image_group, placing same-scene images in train and test
X = df[["brightness", "contrast", "sharpness", "aspect_ratio", "file_size"]].values
y = df["label"].values

# Leakage: StratifiedKFold splits by class, not by image_group (same scene)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=0)
scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
print("CV Accuracy scores:", scores)
print("Mean:", scores.mean())

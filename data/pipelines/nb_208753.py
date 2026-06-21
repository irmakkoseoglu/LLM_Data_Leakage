# Overlap leakage: cross-validation on text embeddings computed from full corpus
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("news_articles.csv")

# Leakage: TF-IDF fitted on all documents — IDF scores are influenced by test set vocab
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["category"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=1000, C=1.0)
scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
print("CV Accuracy:", scores)
print("Mean:", scores.mean())

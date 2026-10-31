"""Forensic Analysis: Discover the exact discriminative markers between
GPT Siblings (gpt-5.5 vs 5.6-luna vs 5.6-sol vs 5.6-terra vs gpt-6-astra).
Strict prompt-paired comparison across official train.csv.
"""
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report

DATA_PATH = r"E:\projects\kaggle\data\official_comp\train.csv"
print("Loading train.csv...")
df = pd.read_csv(DATA_PATH)

# Filter for GPT variants
gpt_models = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra']
gpt_df = df[df["model"].isin(gpt_models)].copy()
print(f"GPT subset: {len(gpt_df)} samples across {gpt_df['prompt_id'].nunique()} prompts")

# 1. PAIRED STATISTICAL DIFFERENCES
print("\n--- PAIRED STATISTICAL COMPARISON ACROSS GPT SIBLINGS ---")
for col_name, func in [
    ("Length (chars)", lambda t: len(t)),
    ("Word count", lambda t: len(t.split())),
    ("Sent count", lambda t: len(re.split(r'[.!?]+', t))),
    ("Colons", lambda t: t.count(":")),
    ("Hyphens/Dashes", lambda t: t.count("-") + t.count("—")),
    ("Bullet points", lambda t: t.count("\n- ") + t.count("\n* ")),
    ("Bold markdown", lambda t: t.count("**")),
    ("Backticks", lambda t: t.count("`")),
    ("Hapax ratio", lambda t: len([w for w, c in Counter(t.lower().split()).items() if c == 1]) / max(len(t.split()), 1)),
]:
    gpt_df[col_name] = gpt_df["text"].fillna("").apply(func)
    print(f"\n{col_name}:")
    for m in gpt_models:
        vals = gpt_df[gpt_df["model"] == m][col_name]
        print(f"  {m:18s} | Mean: {vals.mean():.2f} | Std: {vals.std():.2f} | Median: {vals.median():.2f}")

# 2. DISCRIMINATIVE N-GRAM PROBE
print("\n--- TOP DISCRIMINATIVE CHAR/WORD N-GRAMS FOR GPT SIBLINGS ---")
tfidf = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', min_df=3, max_features=5000)
X_tfidf = tfidf.fit_transform(gpt_df["text"].fillna(""))
y_gpt = gpt_df["model"].values
groups = gpt_df["prompt_id"].values

gkf = GroupKFold(n_splits=5)
oof_preds = []
oof_indices = []

for train_idx, val_idx in gkf.split(X_tfidf, y_gpt, groups):
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(X_tfidf[train_idx], y_gpt[train_idx])
    preds = clf.predict(X_tfidf[val_idx])
    oof_preds.extend(preds)
    oof_indices.extend(val_idx)

oof_preds = np.array(oof_preds)[np.argsort(oof_indices)]
print("\n5-Fold GroupKFold OOF Classification Report (GPT Siblings):")
print(classification_report(y_gpt, oof_preds))

print("\n--- FORENSIC PROBE COMPLETE ---")

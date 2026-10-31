"""Forensic Analysis: Discover the exact discriminative markers between
Gemini Flash Siblings (3.6 vs 3.7 vs 3.8) and GPT Siblings (5.5 vs 5.6-luna vs 5.6-terra).
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

# Filter for Gemini Flash variants
gemini_flash = df[df["model"].isin(["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"])].copy()
print(f"Gemini Flash subset: {len(gemini_flash)} samples across {gemini_flash['prompt_id'].nunique()} prompts")

# 1. PAIRED STATISTICAL DIFFERENCES
print("\n--- PAIRED STATISTICAL COMPARISON ACROSS GEMINI FLASH SIBLINGS ---")
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
    gemini_flash[col_name] = gemini_flash["text"].fillna("").apply(func)
    print(f"\n{col_name}:")
    for m in ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]:
        vals = gemini_flash[gemini_flash["model"] == m][col_name]
        print(f"  {m:18s} | Mean: {vals.mean():.2f} | Std: {vals.std():.2f} | Median: {vals.median():.2f}")

# 2. DISCRIMINATIVE N-GRAM PROBE
print("\n--- TOP DISCRIMINATIVE CHAR/WORD N-GRAMS FOR GEMINI SIBLINGS ---")
tfidf = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', min_df=3, max_features=5000)
X_tfidf = tfidf.fit_transform(gemini_flash["text"].fillna(""))
y_gemini = gemini_flash["model"].values
groups = gemini_flash["prompt_id"].values

gkf = GroupKFold(n_splits=5)
oof_preds = []
oof_indices = []

for train_idx, val_idx in gkf.split(X_tfidf, y_gemini, groups):
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(X_tfidf[train_idx], y_gemini[train_idx])
    preds = clf.predict(X_tfidf[val_idx])
    oof_preds.extend(preds)
    oof_indices.extend(val_idx)

oof_preds = np.array(oof_preds)[np.argsort(oof_indices)]
print("\n5-Fold GroupKFold OOF Classification Report (Gemini Flash Siblings):")
print(classification_report(y_gemini, oof_preds))

# Feature importances
clf_all = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(X_tfidf, y_gemini)
feature_names = np.array(tfidf.get_feature_names_out())
for i, m in enumerate(clf_all.classes_):
    top_indices = np.argsort(clf_all.coef_[i])[-8:][::-1]
    bottom_indices = np.argsort(clf_all.coef_[i])[:5]
    print(f"\nTop positive n-grams for {m}:")
    for idx in top_indices:
        print(f"  {repr(feature_names[idx]):20s} (coef: {clf_all.coef_[i][idx]:+.4f})")

print("\n--- FORENSIC PROBE COMPLETE ---")

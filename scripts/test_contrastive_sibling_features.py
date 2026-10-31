"""Experiment: Contrastive & Differential Stylometrics for Sibling Disambiguation.
Tests whether adding pairwise differential features (relative to family mean)
improves 5-Fold GroupKFold OOF balanced accuracy on official train.csv.
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append(r"E:\projects\kaggle\scripts")
from test_enhanced_sibling_style import extract_features

DATA_PATH = r"E:\projects\kaggle\data\official_comp\train.csv"
print("Loading train.csv...")
df = pd.read_csv(DATA_PATH)
labels = df["model"].tolist()
unique_labels = sorted(list(set(labels)))
label_to_idx = {l: i for i, l in enumerate(unique_labels)}
y = np.array([label_to_idx[l] for l in labels])
groups = df["prompt_id"].values
texts = df["text"].fillna("").astype(str).tolist()

print("Extracting base 163 stylometric features...")
X_base = extract_features(texts)

# Add domain-specific differential features:
# 1. Dash-to-comma ratio
# 2. Sent-length variance to mean ratio (coefficient of variation)
# 3. Uppercase to word ratio
# 4. Question marks to sentence ratio
print("Constructing contrastive differential features...")
diff_features = []
for t in texts:
    t_str = str(t)
    n_words = max(len(t_str.split()), 1)
    n_sents = max(len(re.split(r'[.!?]+', t_str)), 1)
    n_dashes = t_str.count("-") + t_str.count("—")
    n_commas = t_str.count(",")
    n_colons = t_str.count(":")
    n_upper = sum(1 for c in t_str if c.isupper())
    
    diff_row = [
        n_dashes / max(n_commas + 1, 1),
        n_colons / max(n_sents, 1),
        n_upper / n_words,
        (t_str.count("?") + t_str.count("!")) / n_sents,
        len(t_str) / max(n_words, 1), # average word length in chars
    ]
    diff_features.append(diff_row)

X_diff = np.array(diff_features, dtype=np.float32)
X_augmented = np.hstack([X_base, X_diff])
print(f"Augmented feature shape: {X_augmented.shape} (added {X_diff.shape[1]} contrastive features)")

FAMILIES = {
    "CLAUDE": ([m for m in unique_labels if "claude" in m], 0.01),
    "GEMINI": ([m for m in unique_labels if "gemini" in m], 0.01),
    "GPT": ([m for m in unique_labels if "gpt" in m], 0.003),
    "GLM": ([m for m in unique_labels if "glm" in m], 0.10),
}

print("\n--- 5-Fold GroupKFold OOF Comparison: Base vs Augmented ---")
for fam_name, (fam_members, C) in FAMILIES.items():
    cols = np.flatnonzero(np.char.startswith(np.asarray(unique_labels), fam_name.lower()))
    train_mask = np.isin(y, cols)
    inds = np.flatnonzero(train_mask)
    y_fam = np.searchsorted(cols, y[inds])
    grp_fam = groups[inds]
    chance = 1.0 / len(cols)
    
    # 1. Base 163 features
    oof_base = np.empty(len(y_fam), dtype=np.int32)
    for tr, val in GroupKFold(n_splits=5).split(X_base[inds], y_fam, groups=grp_fam):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
        clf.fit(X_base[inds][tr], y_fam[tr])
        oof_base[val] = clf.predict(X_base[inds][val])
    acc_base = balanced_accuracy_score(y_fam, oof_base)
    
    # 2. Augmented features
    oof_aug = np.empty(len(y_fam), dtype=np.int32)
    for tr, val in GroupKFold(n_splits=5).split(X_augmented[inds], y_fam, groups=grp_fam):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
        clf.fit(X_augmented[inds][tr], y_fam[tr])
        oof_aug[val] = clf.predict(X_augmented[inds][val])
    acc_aug = balanced_accuracy_score(y_fam, oof_aug)
    
    delta = acc_aug - acc_base
    print(f"Family {fam_name:8s} | Chance: {chance:.4f} | Base OOF: {acc_base:.4f} | Augmented OOF: {acc_aug:.4f} ({delta:+.4f})")

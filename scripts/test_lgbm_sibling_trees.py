import json
import re
import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
import time

LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max"
]

MARKERS = [
    ',', '.', ';', ':', '!', '?',
    '\u2014', '\u2013', '-', '\u2019', '\u2018', '\u201c', '\u201d', '"', "'",
    '(', ')', '[', ']', '{', '}', '*', '`', '\u2022', '\u2026', '/', '\\',
    '\n', '\n\n', '\n- ', '\n* ', '\n1. ', '\n2. ', '## ', '### ', '```', '**', '> ', '\t',
    ' = ', ' e.g.', ' i.e.', 'However,', 'Overall,', 'In conclusion',
    'Therefore,', 'Moreover,', 'First,', 'Finally,', ' - ', ': ', ' ; ', '  ',
    '|', '---', 'http', 'def ', 'class ', 'import ', 'return ', 'console.log'
]

WORDS = [
    'the', 'and', 'but', 'that', 'which', 'with', 'into', 'from', 'for', 'you',
    'your', 'our', 'we', 'I', 'it', 'its', 'they', 'their', 'there', 'this',
    'these', 'those', 'however', 'therefore', 'moreover', 'although', 'because',
    'perhaps', 'might', 'would', 'could', 'should', 'like', 'such', 'just', 'very',
    'also', 'not', 'if', 'or', 'as', 'so', 'by', 'of', 'to', 'in', 'on', 'at',
    'is', 'are', 'be', 'was', 'were', 'have', 'has', 'had', 'may', 'must', 'all',
    'any', 'each', 'every', 'many', 'some', 'more', 'most', 'only', 'even', 'still',
    'then', 'when', 'where', 'why', 'how', 'what', 'who', 'furthermore', 'additionally',
    'specifically', 'consequently', 'essentially', 'ultimately', 'namely'
]

WRE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?")
SRE = re.compile(r'(?<=[.?!])\s+')

@lru_cache(maxsize=15000)
def extract_row(t):
    t = str(t)
    s = WRE.findall(t)
    lo = [z.lower() for z in s]
    n = max(len(s), 1)
    nt = max(len(t), 1)
    lines = t.splitlines()
    ns = max(len(lines), 1)
    lens = np.array([len(z) for z in s] or [0])
    sent_lens = np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    base = [
        np.log1p(n), np.log1p(nt), nt / n, ns / n, len(set(lo)) / n,
        float(np.mean(lens)), float(np.std(lens)),
        float(np.mean(sent_lens)), float(np.std(sent_lens)), float(np.max(sent_lens)),
        float(np.mean(lens >= 8)), float(np.mean(lens <= 3)),
        sum(c.isupper() for c in t) / nt,
        sum(c.isdigit() for c in t) / nt,
        sum(ord(c) > 127 for c in t) / nt,
        sum(c.isspace() for c in t) / nt,
        sum(c == '\n' for c in t) / nt,
        sum(len(l.strip()) == 0 for l in lines) / ns,
        float(np.mean([len(l) for l in lines] or [0])),
    ]
    base += [min(t.count(m) / n, 3) for m in MARKERS]
    cnt = {}
    for w in lo:
        cnt[w] = cnt.get(w, 0) + 1
    base += [cnt.get(w, 0) / n for w in WORDS]
    return np.array(base, dtype=np.float32)

def extract_features(texts):
    return np.stack([extract_row(str(t)) for t in texts])

def main():
    print("Loading train.csv...")
    df = pd.read_csv(r"E:\projects\kaggle\data\train.csv")
    texts = df['text'].fillna('').astype(str).tolist()
    y = np.array([LABELS.index(m) for m in df['model']])
    groups = df['prompt_id'].values

    print("Extracting 163 stylometric features...")
    X = extract_features(texts)

    families = [
        ('claude', 5),
        ('gemini', 4),
        ('gpt', 5),
        ('glm', 2)
    ]

    print("\n--- Testing LightGBM Trees on Within-Family Sibling Discrimination ---")
    gkf = GroupKFold(n_splits=5)
    lab = np.asarray(LABELS)

    for family, num_classes in families:
        cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))
        mask = np.isin(y, cols)
        inds = np.flatnonzero(mask)
        X_fam = X[inds]
        y_fam = np.searchsorted(cols, y[inds])
        grp_fam = groups[inds]
        chance = 1.0 / len(cols)

        oof_pred = np.empty(len(y_fam), dtype=np.int32)
        params = {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 15,
            'max_depth': 4,
            'feature_fraction': 0.8,
            'verbose': -1,
            'random_state': 42
        }

        for tr, val in gkf.split(X_fam, y_fam, groups=grp_fam):
            trn_data = lgb.Dataset(X_fam[tr], label=y_fam[tr])
            val_data = lgb.Dataset(X_fam[val], label=y_fam[val], reference=trn_data)
            gbm = lgb.train(params, trn_data, num_boost_round=150, valid_sets=[val_data],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
            val_probs = gbm.predict(X_fam[val], num_iteration=gbm.best_iteration)
            oof_pred[val] = val_probs.argmax(axis=1)

        bacc = balanced_accuracy_score(y_fam, oof_pred)
        gain = bacc - chance
        print(f"Family: {family.upper():<7} | Chance: {chance:.4f} | LGBM 5-Fold OOF Acc: {bacc:.4f} (+{gain*100:+.2f}%)")

if __name__ == '__main__':
    main()

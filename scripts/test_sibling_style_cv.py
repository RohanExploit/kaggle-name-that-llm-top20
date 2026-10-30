import re
import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score

MARKERS = [
    ',', '.', ';', ':', '!', '?',
    '\u2014', '\u2013', '-', '\u2019', '\u2018', '\u201c', '\u201d', '"', "'",
    '(', ')', '[', ']', '{', '}', '*', '`', '\u2022', '\u2026', '/', '\\', ':',
    '\n', '\n\n', '\n-', '\n1.', '\n2.', '##', '```', '**', '> ', '\t',
    ' = ', ' e.g.', ' i.e.', 'However,', 'Overall,', 'In conclusion',
    'Therefore,', 'Moreover,', 'First,', 'Finally,', ' - ', ': ', ' ; ', '  '
]

WORDS = [
    'the', 'and', 'but', 'that', 'which', 'with', 'into', 'from', 'for', 'you',
    'your', 'our', 'we', 'I', 'it', 'its', 'they', 'their', 'there', 'this',
    'these', 'those', 'however', 'therefore', 'moreover', 'although', 'because',
    'perhaps', 'might', 'would', 'could', 'should', 'like', 'such', 'just', 'very',
    'also', 'not', 'if', 'or', 'as', 'so', 'by', 'of', 'to', 'in', 'on', 'at',
    'is', 'are', 'be', 'was', 'were', 'have', 'has', 'had', 'may', 'must', 'all',
    'any', 'each', 'every', 'many', 'some', 'more', 'most', 'only', 'even', 'still',
    'then', 'when', 'where', 'why', 'how', 'what', 'who'
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
    ns = max(len(t.splitlines()), 1)
    lens = np.array([len(z) for z in s] or [0])
    sent_lens = np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    base = [
        np.log1p(n), np.log1p(nt), nt / n, ns / n, len(set(lo)) / n,
        float(np.mean(lens)), float(np.std(lens)),
        float(np.mean(sent_lens)), float(np.std(sent_lens)), float(np.max(sent_lens)),
        float(np.mean(lens >= 8)),
        sum(c.isupper() for c in t) / nt,
        sum(c.isdigit() for c in t) / nt,
        sum(ord(c) > 127 for c in t) / nt,
        sum(c.isspace() for c in t) / nt,
        sum(c == '\n' for c in t) / nt,
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
    print("Loading official train.csv...")
    df = pd.read_csv(r"E:\projects\kaggle\data\train.csv")
    print(f"Loaded {len(df)} rows across {df['model'].nunique()} models.")

    print("Extracting stylometric features...")
    X = extract_features(df['text'].tolist())
    print(f"Feature matrix shape: {X.shape}")

    labels = np.array(sorted(df['model'].unique()))
    y = np.array([np.where(labels == m)[0][0] for m in df['model']])
    groups = df['prompt_id'].values

    families = [
        ('claude', 0.003),
        ('gemini', 0.003),
        ('gpt', 0.003),
        ('glm', 0.03)
    ]

    print("\n--- 5-Fold GroupKFold Within-Family Sibling Stylometry CV ---")
    for family, C in families:
        cols = np.flatnonzero(np.char.startswith(labels.astype(str), family))
        if len(cols) < 2:
            continue
        mask = np.isin(y, cols)
        inds = np.flatnonzero(mask)
        X_fam = X[inds]
        y_fam = np.searchsorted(cols, y[inds])
        grp_fam = groups[inds]

        oof_pred = np.empty(len(y_fam), dtype=np.int32)
        gkf = GroupKFold(n_splits=5)
        for train_idx, val_idx in gkf.split(X_fam, y_fam, groups=grp_fam):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=C, max_iter=400, class_weight='balanced')
            )
            clf.fit(X_fam[train_idx], y_fam[train_idx])
            oof_pred[val_idx] = clf.predict(X_fam[val_idx])

        bacc = balanced_accuracy_score(y_fam, oof_pred)
        chance = 1.0 / len(cols)
        gain = bacc - chance
        reliability = np.clip(gain / 0.15, 0.0, 1.0)
        fam_models = labels[cols].tolist()

        print(f"\nFamily: {family.upper()} (models: {fam_models})")
        print(f"  Random Chance Level: {chance:.4f} ({1/chance:.0f} models)")
        print(f"  5-Fold GroupKFold OOF Balanced Acc: {bacc:.4f} (+{gain*100:+.2f}% over chance)")
        print(f"  Calculated Reliability: {reliability:.4f}")

if __name__ == '__main__':
    main()

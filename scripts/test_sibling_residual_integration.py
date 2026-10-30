import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
import time

LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max"
]

import re
from functools import lru_cache

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

def apply_grouped_style_residual(logits, train_texts, test_texts, labels, y, groups, max_alpha=0.08):
    """OOF-estimated conservative sibling re-ranking. Total family probability mass is mathematically conserved."""
    Xt = extract_features(train_texts)
    Xq = extract_features(test_texts)
    out = np.asarray(logits, dtype=np.float64).copy()
    lab = np.asarray(labels)
    yy = np.asarray(y)
    g = np.asarray(groups)

    family_configs = [
        ('claude', 0.01),
        ('gemini', 0.01),
        ('gpt', 0.003),
        ('glm', 0.1)
    ]

    for family, C in family_configs:
        cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))
        if len(cols) < 2:
            continue
        train_mask = np.isin(yy, cols)
        inds = np.flatnonzero(train_mask)
        X = Xt[inds]
        target = np.searchsorted(cols, yy[inds])
        grp = g[inds]

        oof = np.empty(len(target), dtype=np.int32)
        for tr, val in GroupKFold(n_splits=5).split(X, target, groups=grp):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
            model.fit(X[tr], target[tr])
            oof[val] = model.predict(X[val])

        bacc = float(balanced_accuracy_score(target, oof))
        chance = 1.0 / len(cols)
        reliability = np.clip((bacc - chance) / 0.15, 0.0, 1.0)
        if reliability <= 0:
            continue

        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
        model.fit(X, target)
        style_p = np.maximum(model.predict_proba(Xq), 1e-8)

        old_logits = out[:, cols].copy()
        old_p = softmax(old_logits, axis=1)
        margin = np.partition(old_p, -2, axis=1)[:, -1] - np.partition(old_p, -2, axis=1)[:, -2]
        gate = np.clip((0.65 - margin) / 0.65, 0.0, 1.0)
        strength = (max_alpha * reliability) * gate
        mix = (1.0 - strength[:, None]) * old_p + strength[:, None] * style_p

        out[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))
        assert np.allclose(logsumexp(out[:, cols], axis=1), logsumexp(old_logits, axis=1), atol=1e-8)

    return out

def main():
    print("Testing integration...")
    train_df = pd.read_csv(r"E:\projects\kaggle\data\train.csv")
    train_texts = train_df['text'].tolist()
    y = np.array([LABELS.index(m) for m in train_df['model']])
    groups = train_df['prompt_id'].values

    # Test on a simulated batch of 500 test texts
    test_texts = train_texts[:500]
    # Create fake neural logits
    np.random.seed(42)
    fake_logits = np.random.randn(500, 18)

    t0 = time.time()
    refined_logits = apply_grouped_style_residual(fake_logits, train_texts, test_texts, LABELS, y, groups)
    dt = time.time() - t0

    print(f"Success! Runtime for 500 passages: {dt:.2f}s")
    print(f"Logits shape: {refined_logits.shape}")
    print(f"Finite check: {np.isfinite(refined_logits).all()}")

if __name__ == '__main__':
    main()

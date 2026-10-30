import json
import re
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.sparse import hstack
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
import time

LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max"
]

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())

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

    print(f"Loaded {len(df)} rows. Extracting stylometric features...")
    X_style = extract_features(texts)

    print("Building Char + Word TF-IDF feature matrices...")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=40000, sublinear_tf=True)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)

    gkf = GroupKFold(n_splits=5)
    oof_preds = np.empty((len(y), len(LABELS)), dtype=np.float64)

    t0 = time.time()
    for fold, (tr, val) in enumerate(gkf.split(texts, y, groups=groups)):
        print(f"Evaluating Fold {fold+1}/5...")
        # Fit TF-IDF only on training fold to guarantee zero data leakage
        tr_texts = [texts[i] for i in tr]
        val_texts = [texts[i] for i in val]

        X_tr_char = char_vec.fit_transform(tr_texts)
        X_val_char = char_vec.transform(val_texts)

        X_tr_word = word_vec.fit_transform(tr_texts)
        X_val_word = word_vec.transform(val_texts)

        # Scale stylometrics using train fold statistics
        scaler = StandardScaler()
        X_tr_sty = scaler.fit_transform(X_style[tr])
        X_val_sty = scaler.transform(X_style[val])

        # Stack views
        X_tr = hstack([X_tr_char * 1.0, X_tr_word * 0.7, X_tr_sty * 0.5], format="csr")
        X_val = hstack([X_val_char * 1.0, X_val_word * 0.7, X_val_sty * 0.5], format="csr")

        clf = LogisticRegression(C=2.0, max_iter=600, class_weight='balanced')
        clf.fit(X_tr, y[tr])
        oof_preds[val] = clf.predict_proba(X_val)

    dt = time.time() - t0
    final_pred = oof_preds.argmax(axis=1)
    bacc = balanced_accuracy_score(y, final_pred)

    print("\n" + "="*60)
    print(f"5-Fold Strict GroupKFold Zero-Leakage Balanced Acc: {bacc:.4f}")
    print(f"Elapsed Time: {dt:.2f}s")
    print("="*60)

    # Per-class accuracy
    print("\nPer-Class Breakdown:")
    for i, lab in enumerate(LABELS):
        mask = (y == i)
        acc = (final_pred[mask] == i).mean()
        print(f"  {lab:<25}: {acc:.4f}")

if __name__ == '__main__':
    main()

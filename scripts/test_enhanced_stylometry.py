import re
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from src.data import OFFICIAL_18_CLASSES, LABEL_TO_ID, create_prompt_grouped_folds

train_df = pd.read_csv("data/train.csv")
train_df = train_df.dropna(subset=["text", "model", "prompt_id"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
train_df["label_id"] = train_df["model"].map(LABEL_TO_ID)

# Baseline stylometry
def baseline_stylometry(texts):
    rows = []
    for t in texts:
        words = t.split()
        n_w = len(words) if words else 1
        sents = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        n_s = len(sents) if sents else 1
        rows.append([
            1.0 if '—' in t else 0.0,
            t.count('—') / n_w,
            t.count('’') / (t.count("'") + t.count('’') + 1e-5),
            t.count(';') / n_w,
            t.count(':') / n_w,
            len(set(words)) / n_w,
            np.mean([len(w) for w in words]) if words else 0.0,
            n_w / n_s,
        ])
    return np.array(rows, dtype=np.float32)

# Enhanced structural & token forensics
def enhanced_stylometry(texts):
    rows = []
    for t in texts:
        words = t.split()
        n_w = len(words) if words else 1
        lines = [line.strip() for line in t.split('\n') if line.strip()]
        n_lines = len(lines) if lines else 1
        sents = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        n_s = len(sents) if sents else 1
        
        # Punctuation & Quotes
        has_em_dash = 1.0 if '—' in t else 0.0
        em_dash_cnt = t.count('—') / n_w
        curly_apostrophe = t.count('’') / (t.count("'") + t.count('’') + 1e-5)
        curly_quote = (t.count('“') + t.count('”')) / (t.count('"') + t.count('“') + t.count('”') + 1e-5)
        semicolons = t.count(';') / n_w
        colons = t.count(':') / n_w
        exclamations = t.count('!') / n_w
        parentheses = t.count('(') / n_w
        
        # Lexical & Structural
        ttr = len(set(words)) / n_w
        avg_w_len = np.mean([len(w) for w in words]) if words else 0.0
        avg_s_len = n_w / n_s
        newline_density = t.count('\n') / n_w
        uppercase_ratio = sum(1 for c in t if c.isupper()) / (len(t) + 1e-5)
        
        # Markdown & Formatting Signatures
        bullet_ratio = sum(1 for l in lines if l.startswith(('- ', '* ', '+ '))) / n_lines
        numbered_ratio = sum(1 for l in lines if re.match(r'^\d+\.\s', l)) / n_lines
        header_ratio = sum(1 for l in lines if l.startswith('#')) / n_lines
        bold_ratio = len(re.findall(r'\*\*.*?\*\*', t)) / n_lines
        code_fence_cnt = t.count('```') / n_lines
        
        rows.append([
            has_em_dash, em_dash_cnt, curly_apostrophe, curly_quote,
            semicolons, colons, exclamations, parentheses,
            ttr, avg_w_len, avg_s_len, newline_density, uppercase_ratio,
            bullet_ratio, numbered_ratio, header_ratio, bold_ratio, code_fence_cnt
        ])
    return np.array(rows, dtype=np.float32)

folds_df = create_prompt_grouped_folds(train_df, n_splits=5, seed=42)

for name, feat_func in [("Baseline (8 features)", baseline_stylometry), ("Enhanced (18 features)", enhanced_stylometry)]:
    oof_preds = np.zeros(len(folds_df))
    for fold in range(5):
        tr = folds_df[folds_df["fold"] != fold]
        va = folds_df[folds_df["fold"] == fold]
        
        word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)
        char_vec = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', min_df=3, max_features=10000, sublinear_tf=True)
        
        X_tr = hstack([word_vec.fit_transform(tr['text']), char_vec.fit_transform(tr['text']), csr_matrix(feat_func(tr['text']))])
        X_va = hstack([word_vec.transform(va['text']), char_vec.transform(va['text']), csr_matrix(feat_func(va['text']))])
        
        clf = LogisticRegression(C=2.5, max_iter=350, class_weight='balanced', solver='lbfgs', random_state=42)
        clf.fit(X_tr, tr['label_id'])
        oof_preds[folds_df["fold"] == fold] = clf.predict(X_va)
        
    acc = balanced_accuracy_score(folds_df['label_id'], oof_preds)
    print(f"[{name}] 5-Fold Prompt-Grouped CV Balanced Acc: {acc:.4f}")

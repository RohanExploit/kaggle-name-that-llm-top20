"""
Find Optimal Alpha Blending Weight for C=4.0 Stylometry + DeBERTa-v3 Fold 2
"""
import os, sys, re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

OFFICIAL_18_CLASSES = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
LABEL_TO_ID = {name: i for i, name in enumerate(OFFICIAL_18_CLASSES)}

def extract_dense_stylometry(texts):
    rows = []
    for t in texts:
        words = t.split()
        n_w = len(words) if words else 1
        sents = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        n_s = len(sents) if sents else 1
        
        has_em_dash = 1.0 if '—' in t else 0.0
        em_dash_cnt = t.count('—') / n_w
        curly_apostrophe = t.count('’') / (t.count("'") + t.count('’') + 1e-5)
        semicolons = t.count(';') / n_w
        colons = t.count(':') / n_w
        ttr = len(set(words)) / n_w
        avg_w_len = np.mean([len(w) for w in words]) if words else 0.0
        avg_s_len = n_w / n_s
        
        rows.append([
            has_em_dash, em_dash_cnt, curly_apostrophe,
            semicolons, colons, ttr, avg_w_len, avg_s_len
        ])
    return np.array(rows, dtype=np.float32)

def main():
    df = pd.read_csv('data/train.csv')
    df = df[df['model'].isin(OFFICIAL_18_CLASSES)].dropna(subset=['text', 'prompt_id']).reset_index(drop=True)
    df['label_id'] = df['model'].map(LABEL_TO_ID)
    y_true = df['label_id'].values
    
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    df['fold'] = -1
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(df, y=df['label_id'], groups=df['prompt_id'])):
        df.loc[val_idx, 'fold'] = fold
        
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)
    char_vec = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', min_df=3, max_features=10000, sublinear_tf=True)
    
    neural_oof = np.load('checkpoints/gpu_transformer/oof_probs.npy')[:len(df)]
    sty_oof = np.zeros((len(df), 18), dtype=np.float32)
    
    for fold in range(3):
        tr_df = df[df['fold'] != fold].reset_index(drop=True)
        val_df = df[df['fold'] == fold].reset_index(drop=True)
        
        X_tr_w = word_vec.fit_transform(tr_df['text'])
        X_tr_c = char_vec.fit_transform(tr_df['text'])
        X_tr_s = csr_matrix(extract_dense_stylometry(tr_df['text']))
        X_tr = hstack([X_tr_w, X_tr_c, X_tr_s])
        
        X_val_w = word_vec.transform(val_df['text'])
        X_val_c = char_vec.transform(val_df['text'])
        X_val_s = csr_matrix(extract_dense_stylometry(val_df['text']))
        X_val = hstack([X_val_w, X_val_c, X_val_s])
        
        clf = LogisticRegression(C=4.0, max_iter=300, class_weight='balanced', solver='lbfgs', random_state=42)
        clf.fit(X_tr, tr_df['label_id'])
        sty_oof[df['fold'] == fold] = clf.predict_proba(X_val)
        
    print(f"C=4.0 Stylometry OOF Score: {balanced_accuracy_score(y_true, np.argmax(sty_oof, axis=1)):.4f}")
    
    for alpha in [0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]:
        blend = alpha * neural_oof + (1.0 - alpha) * sty_oof
        score = balanced_accuracy_score(y_true, np.argmax(blend, axis=1))
        print(f"  Alpha (Neural) = {alpha:.2f} | Balanced Acc: {score:.4f}")

if __name__ == '__main__':
    main()

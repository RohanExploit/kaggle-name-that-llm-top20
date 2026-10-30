"""
Evaluate Ensemble Probability Blending
Optimizes alpha blending between 3-Fold Neural DeBERTa-v3 and Stylometric Forensics.
"""
import os, glob, re
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from scipy.sparse import hstack, csr_matrix

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
        sents = re.split(r'[.!?]+', t)
        sents = [s.strip() for s in sents if s.strip()]
        n_s = len(sents) if sents else 1
        
        has_em_dash = 1.0 if '—' in t else 0.0
        em_dash_cnt = t.count('—') / n_w
        curly_apostrophe = t.count('’') / (t.count("'") + t.count('’') + 1e-5)
        semicolons = t.count(';') / n_w
        colons = t.count(':') / n_w
        ttr = len(set(words)) / n_w
        avg_w_len = np.mean([len(w) for w in words]) if words else 0.0
        avg_s_len = n_w / n_s
        rows.append([has_em_dash, em_dash_cnt, curly_apostrophe, semicolons, colons, ttr, avg_w_len, avg_s_len])
    return np.array(rows, dtype=np.float32)

def main():
    # 1. Load official train data
    df = pd.read_csv('data/train.csv')
    df = df[df['model'].isin(OFFICIAL_18_CLASSES)].dropna(subset=['text', 'prompt_id']).reset_index(drop=True)
    df['label_id'] = df['model'].map(LABEL_TO_ID)
    
    # 2. Recreate same 3 StratifiedGroupKFold on prompt_id
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    df['fold'] = -1
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(df, y=df['label_id'], groups=df['prompt_id'])):
        df.loc[val_idx, 'fold'] = fold
        
    # 3. Load neural OOF probabilities
    neural_oof_path = 'checkpoints/gpu_transformer/oof_probs.npy'
    if not os.path.exists(neural_oof_path):
        print(f"Error: {neural_oof_path} not found")
        return
    
    # Note: neural runner trained on official + balanced extra samples.
    # We evaluate stylometry OOF probabilities strictly on the official folds:
    sty_oof_probs = np.zeros((len(df), 18), dtype=np.float32)
    
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)
    char_vec = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', min_df=3, max_features=10000, sublinear_tf=True)
    
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
        
        clf = LogisticRegression(C=2.5, max_iter=300, class_weight='balanced', solver='lbfgs', random_state=42)
        clf.fit(X_tr, tr_df['label_id'])
        
        val_probs = clf.predict_proba(X_val)
        sty_oof_probs[df['fold'] == fold] = val_probs

    sty_score = balanced_accuracy_score(df['label_id'], np.argmax(sty_oof_probs, axis=1))
    print(f"Stylometric TF-IDF 3-Fold OOF Balanced Acc: {sty_score:.4f}")
    
    # Save stylometry OOF
    np.save('checkpoints/sty_oof_probs.npy', sty_oof_probs)
    print("Saved stylometry OOF to checkpoints/sty_oof_probs.npy")

if __name__ == '__main__':
    main()

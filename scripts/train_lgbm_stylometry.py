"""
Train Non-Linear LightGBM Classifier on Dense Stylometric & SVD Forensics
Evaluates 3-way Tri-Ensemble: DeBERTa-v3 + Logistic Regression + LightGBM
"""
import os, sys, re, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb

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
        lines = [line.strip() for line in t.split('\n') if line.strip()]
        n_lines = len(lines) if lines else 1
        sents = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        n_s = len(sents) if sents else 1
        
        has_em_dash = 1.0 if '—' in t else 0.0
        em_dash_cnt = t.count('—') / n_w
        curly_apostrophe = t.count('’') / (t.count("'") + t.count('’') + 1e-5)
        curly_quote = (t.count('“') + t.count('”')) / (t.count('"') + t.count('“') + t.count('”') + 1e-5)
        semicolons = t.count(';') / n_w
        colons = t.count(':') / n_w
        exclamations = t.count('!') / n_w
        parentheses = t.count('(') / n_w
        
        ttr = len(set(words)) / n_w
        avg_w_len = np.mean([len(w) for w in words]) if words else 0.0
        avg_s_len = n_w / n_s
        newline_density = t.count('\n') / n_w
        uppercase_ratio = sum(1 for c in t if c.isupper()) / (len(t) + 1e-5)
        
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

def main():
    print("Loading data...")
    df = pd.read_csv('data/train.csv')
    df = df[df['model'].isin(OFFICIAL_18_CLASSES)].dropna(subset=['text', 'prompt_id']).reset_index(drop=True)
    df['label_id'] = df['model'].map(LABEL_TO_ID)
    
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    df['fold'] = -1
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(df, y=df['label_id'], groups=df['prompt_id'])):
        df.loc[val_idx, 'fold'] = fold
        
    print(f"Total training samples: {len(df):,}")
    
    # 1. Feature extraction: Stylometry + SVD of TF-IDF
    print("Extracting stylometric features...")
    sty_features = extract_dense_stylometry(df['text'])
    
    print("Extracting SVD TF-IDF features...")
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000, sublinear_tf=True)
    char_vec = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', min_df=3, max_features=8000, sublinear_tf=True)
    
    X_word = word_vec.fit_transform(df['text'])
    X_char = char_vec.fit_transform(df['text'])
    
    svd_word = TruncatedSVD(n_components=40, random_state=42)
    svd_char = TruncatedSVD(n_components=40, random_state=42)
    
    X_svd_w = svd_word.fit_transform(X_word)
    X_svd_c = svd_char.fit_transform(X_char)
    
    X_all = np.hstack([sty_features, X_svd_w, X_svd_c])
    print(f"Dense Feature Matrix shape: {X_all.shape}")
    
    lgb_oof_probs = np.zeros((len(df), 18), dtype=np.float32)
    
    params = {
        'objective': 'multiclass',
        'num_class': 18,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 15,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    for fold in range(3):
        tr_mask = (df['fold'] != fold)
        val_mask = (df['fold'] == fold)
        
        X_tr, y_tr = X_all[tr_mask], df.loc[tr_mask, 'label_id'].values
        X_val, y_val = X_all[val_mask], df.loc[val_mask, 'label_id'].values
        
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)
        
        clf = lgb.train(
            params,
            trn_data,
            num_boost_round=300,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        val_preds = clf.predict(X_val, num_iteration=clf.best_iteration)
        lgb_oof_probs[val_mask] = val_preds
        f_score = balanced_accuracy_score(y_val, np.argmax(val_preds, axis=1))
        print(f"  Fold {fold} LightGBM Balanced Acc: {f_score:.4f} (best iter: {clf.best_iteration})")
        
    overall_lgb_score = balanced_accuracy_score(df['label_id'], np.argmax(lgb_oof_probs, axis=1))
    print(f"Overall LightGBM 3-Fold OOF Balanced Acc: {overall_lgb_score:.4f}")
    
    np.save('checkpoints/lgbm_oof_probs.npy', lgb_oof_probs)
    print("Saved LightGBM OOF to checkpoints/lgbm_oof_probs.npy")
    
    # 2. Evaluate Tri-Ensemble Blend
    if os.path.exists('checkpoints/sty_oof_probs.npy') and os.path.exists('checkpoints/gpu_transformer/oof_probs.npy'):
        sty_oof = np.load('checkpoints/sty_oof_probs.npy')
        neural_oof = np.load('checkpoints/gpu_transformer/oof_probs.npy')[:len(df)]
        y_true = df['label_id'].values
        
        print("\n--- Evaluating Tri-Ensemble Combinations ---")
        score_sty = balanced_accuracy_score(y_true, np.argmax(sty_oof, axis=1))
        score_neu = balanced_accuracy_score(y_true, np.argmax(neural_oof, axis=1))
        score_5050 = balanced_accuracy_score(y_true, np.argmax(0.5 * neural_oof + 0.5 * sty_oof, axis=1))
        print(f"  Stylometry (LR) OOF:    {score_sty:.4f}")
        print(f"  Neural (DeBERTa) OOF:   {score_neu:.4f}")
        print(f"  LightGBM GBDT OOF:      {overall_lgb_score:.4f}")
        print(f"  Current 50/50 Dual:     {score_5050:.4f}")
        
        best_blend = 0.0
        best_weights = None
        for w_neu in np.linspace(0.3, 0.6, 7):
            for w_sty in np.linspace(0.2, 0.5, 7):
                w_lgb = 1.0 - w_neu - w_sty
                if w_lgb < 0.05 or w_lgb > 0.4:
                    continue
                tri_blend = w_neu * neural_oof + w_sty * sty_oof + w_lgb * lgb_oof_probs
                s = balanced_accuracy_score(y_true, np.argmax(tri_blend, axis=1))
                if s > best_blend:
                    best_blend = s
                    best_weights = (w_neu, w_sty, w_lgb)
                    
        print(f"  Optimal Tri-Ensemble Balanced Acc: {best_blend:.4f}")
        print(f"  Optimal Weights: Neural={best_weights[0]:.2f}, Stylometry={best_weights[1]:.2f}, LightGBM={best_weights[2]:.2f}")

if __name__ == '__main__':
    main()

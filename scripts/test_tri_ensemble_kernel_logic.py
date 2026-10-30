"""
Local Verification of Tri-Ensemble Kernel Logic
Tests that DeBERTa-v3 Model Soup + Stylometric LR + LightGBM GBDT
run end-to-end without error and output a 100% valid submission.parquet.
"""
import os, sys, glob, re, time
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OFFICIAL_18_CLASSES = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
LABEL_TO_ID = {name: i for i, name in enumerate(OFFICIAL_18_CLASSES)}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    t0 = time.time()
    train_df = pd.read_csv('data/train.csv')
    train_df = train_df[train_df['model'].isin(OFFICIAL_18_CLASSES)].dropna(subset=['text']).reset_index(drop=True)
    test_df = pd.read_csv('data/test.csv')
    train_df['label_id'] = train_df['model'].map(LABEL_TO_ID)
    print(f"Loaded Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    # 1. Stylometric TF-IDF (LR)
    print("1. Fitting Logistic Regression Forensics...")
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)
    char_vec = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', min_df=3, max_features=10000, sublinear_tf=True)
    
    X_tr_w = word_vec.fit_transform(train_df['text'])
    X_tr_c = char_vec.fit_transform(train_df['text'])
    sty_tr = extract_dense_stylometry(train_df['text'])
    X_train_lr = hstack([X_tr_w, X_tr_c, csr_matrix(sty_tr)])
    
    X_te_w = word_vec.transform(test_df['text'])
    X_te_c = char_vec.transform(test_df['text'])
    sty_te = extract_dense_stylometry(test_df['text'])
    X_test_lr = hstack([X_te_w, X_te_c, csr_matrix(sty_te)])
    
    clf_lr = LogisticRegression(C=2.5, max_iter=300, class_weight='balanced', solver='lbfgs', random_state=42)
    clf_lr.fit(X_train_lr, train_df['label_id'])
    probs_lr = clf_lr.predict_proba(X_test_lr)
    print(f"   LR Test Probs: {probs_lr.shape}")
    
    # 2. LightGBM GBDT Forensics
    print("2. Fitting LightGBM Non-Linear Trees...")
    svd_w = TruncatedSVD(n_components=40, random_state=42)
    svd_c = TruncatedSVD(n_components=40, random_state=42)
    
    X_tr_svd_w = svd_w.fit_transform(X_tr_w)
    X_tr_svd_c = svd_c.fit_transform(X_tr_c)
    X_train_lgb = np.hstack([sty_tr, X_tr_svd_w, X_tr_svd_c])
    
    X_te_svd_w = svd_w.transform(X_te_w)
    X_te_svd_c = svd_c.transform(X_te_c)
    X_test_lgb = np.hstack([sty_te, X_te_svd_w, X_te_svd_c])
    
    trn_data = lgb.Dataset(X_train_lgb, label=train_df['label_id'].values)
    lgb_params = {
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
    clf_lgb = lgb.train(lgb_params, trn_data, num_boost_round=80)
    probs_lgb = clf_lgb.predict(X_test_lgb)
    print(f"   LightGBM Test Probs: {probs_lgb.shape}")
    
    # 3. DeBERTa-v3 Model Soup
    print("3. Loading DeBERTa-v3 Model Soup...")
    m_dir = 'checkpoints/soup_model'
    tokenizer = AutoTokenizer.from_pretrained(m_dir)
    model = AutoModelForSequenceClassification.from_pretrained(m_dir, num_labels=18).to(DEVICE)
    model.eval()
    
    class TestDS(Dataset):
        def __init__(self, texts, tok, max_len=384):
            self.texts = list(texts)
            self.tok = tok
            self.max_len = max_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tok(str(self.texts[idx]), max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0)}
            
    loader = DataLoader(TestDS(test_df['text'], tokenizer, 384), batch_size=8, shuffle=False)
    probs_neu = []
    with torch.no_grad():
        for b in loader:
            ids, mask = b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE)
            out = model(ids, attention_mask=mask)
            p = torch.softmax(out.logits, dim=-1)
            probs_neu.extend(p.cpu().numpy())
    probs_neu = np.array(probs_neu)
    print(f"   Neural Test Probs: {probs_neu.shape}")
    
    # 4. Tri-Ensemble Fusion
    final_probs = 0.55 * probs_neu + 0.35 * probs_lr + 0.10 * probs_lgb
    pred_ids = np.argmax(final_probs, axis=1)
    predicted_models = [OFFICIAL_18_CLASSES[i] for i in pred_ids]
    
    sub = pd.DataFrame({'id': test_df['id'], 'model': predicted_models})
    assert len(sub) == len(test_df)
    assert not sub['model'].isnull().any()
    assert set(sub['model']).issubset(set(OFFICIAL_18_CLASSES))
    
    os.makedirs('outputs', exist_ok=True)
    sub.to_parquet('outputs/tri_ensemble_submission.parquet', index=False)
    print(f"SUCCESS! Tri-Ensemble verified in {time.time() - t0:.2f}s!")
    print(sub.head())

if __name__ == '__main__':
    main()

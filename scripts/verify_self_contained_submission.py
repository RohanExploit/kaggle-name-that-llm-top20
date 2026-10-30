"""
Verify Self-Contained Kaggle Submission Kernel Logic
Tests that a standalone script can train on train.csv and predict on test.csv
with zero external weights, generating a 100% valid submission.parquet in < 5 seconds.
"""
import os, glob, time, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix

OFFICIAL_18_CLASSES = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]

def extract_dense_stylometry(texts):
    rows = []
    for t in texts:
        l = len(t) if len(t) > 0 else 1
        words = t.split()
        n_w = len(words) if len(words) > 0 else 1
        sents = re.split(r'[.!?]+', t)
        sents = [s.strip() for s in sents if s.strip()]
        n_s = len(sents) if len(sents) > 0 else 1
        
        # Punctuation markers with extreme model discrimination
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

def run_self_contained_pipeline():
    t0 = time.time()
    
    # Locate train and test
    train_hits = glob.glob('/kaggle/input/**/train.csv', recursive=True) or ['data/train.csv', 'data/official_comp/train.csv']
    test_hits = glob.glob('/kaggle/input/**/test.csv', recursive=True) or ['data/test.csv', 'data/official_comp/test.csv']
    
    train_df = pd.read_csv(train_hits[0])
    test_df = pd.read_csv(test_hits[0])
    print(f"Loaded Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    # Vectorizers
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)
    char_vec = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', min_df=3, max_features=10000, sublinear_tf=True)
    
    print("Fitting vectorizers...")
    X_train_word = word_vec.fit_transform(train_df['text'])
    X_train_char = char_vec.fit_transform(train_df['text'])
    X_train_sty = csr_matrix(extract_dense_stylometry(train_df['text']))
    X_train = hstack([X_train_word, X_train_char, X_train_sty])
    
    X_test_word = word_vec.transform(test_df['text'])
    X_test_char = char_vec.transform(test_df['text'])
    X_test_sty = csr_matrix(extract_dense_stylometry(test_df['text']))
    X_test = hstack([X_test_word, X_test_char, X_test_sty])
    
    # Classifier
    print("Training classifier...")
    clf = LogisticRegression(C=2.5, max_iter=300, class_weight='balanced', solver='saga', n_jobs=-1, random_state=42)
    clf.fit(X_train, train_df['model'])
    
    preds = clf.predict(X_test)
    
    sub = pd.DataFrame({'id': test_df['id'], 'model': preds})
    
    # Assertions
    assert len(sub) == len(test_df), f"Expected {len(test_df)} rows, got {len(sub)}"
    assert not sub['model'].isnull().any(), "Null values detected"
    assert set(sub['model']).issubset(set(OFFICIAL_18_CLASSES)), "Invalid labels present"
    
    output_path = 'submission.parquet'
    sub.to_parquet(output_path, index=False)
    
    elapsed = time.time() - t0
    print(f"SUCCESS! Created {output_path} in {elapsed:.2f} seconds.")
    print(sub.head())
    print(f"Predictions distribution:\n{sub['model'].value_counts()}")

if __name__ == '__main__':
    run_self_contained_pipeline()

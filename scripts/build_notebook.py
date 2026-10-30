import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Name That LLM: Autonomous Ensemble Submission Kernel\n",
                "### Kaggle Code Competition Submission (Internet Disabled, Self-Contained, Runtime < 1 min)\n",
                "\n",
                "Features:\n",
                "1. Works completely offline with zero external network access.\n",
                "2. Extracts deep stylometric fingerprints (em-dashes, curly quotes, semicolon rates, TTR).\n",
                "3. Blends TF-IDF n-grams with punctuation forensics.\n",
                "4. Validates output strictly against official 18 model classes and writes `submission.parquet`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob, re, time\n",
                "import numpy as np, pandas as pd\n",
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from scipy.sparse import hstack, csr_matrix\n",
                "\n",
                "OFFICIAL_18_CLASSES = [\n",
                "    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',\n",
                "    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',\n",
                "    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',\n",
                "    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'\n",
                "]\n",
                "\n",
                "def locate_file(pattern, fallback):\n",
                "    hits = glob.glob(pattern, recursive=True)\n",
                "    return hits[0] if hits else fallback\n",
                "\n",
                "test_path = locate_file('/kaggle/input/**/test.csv', 'data/test.csv')\n",
                "train_path = locate_file('/kaggle/input/**/train.csv', 'data/train.csv')\n",
                "print(f'Reading train: {train_path}')\n",
                "print(f'Reading test:  {test_path}')\n",
                "\n",
                "train_df = pd.read_csv(train_path)\n",
                "test_df = pd.read_csv(test_path)\n",
                "print(f'Train size: {len(train_df)} | Test size: {len(test_df)}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def extract_dense_stylometry(texts):\n",
                "    rows = []\n",
                "    for t in texts:\n",
                "        words = t.split()\n",
                "        n_w = len(words) if words else 1\n",
                "        sents = re.split(r'[.!?]+', t)\n",
                "        sents = [s.strip() for s in sents if s.strip()]\n",
                "        n_s = len(sents) if sents else 1\n",
                "        \n",
                "        has_em_dash = 1.0 if '—' in t else 0.0\n",
                "        em_dash_cnt = t.count('—') / n_w\n",
                "        curly_apostrophe = t.count('’') / (t.count(\"'\") + t.count('’') + 1e-5)\n",
                "        semicolons = t.count(';') / n_w\n",
                "        colons = t.count(':') / n_w\n",
                "        ttr = len(set(words)) / n_w\n",
                "        avg_w_len = np.mean([len(w) for w in words]) if words else 0.0\n",
                "        avg_s_len = n_w / n_s\n",
                "        \n",
                "        rows.append([has_em_dash, em_dash_cnt, curly_apostrophe, semicolons, colons, ttr, avg_w_len, avg_s_len])\n",
                "    return np.array(rows, dtype=np.float32)\n",
                "\n",
                "print('Building multi-view representation...')\n",
                "word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)\n",
                "char_vec = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', min_df=3, max_features=10000, sublinear_tf=True)\n",
                "\n",
                "X_tr_w = word_vec.fit_transform(train_df['text'])\n",
                "X_tr_c = char_vec.fit_transform(train_df['text'])\n",
                "X_tr_s = csr_matrix(extract_dense_stylometry(train_df['text']))\n",
                "X_train = hstack([X_tr_w, X_tr_c, X_tr_s])\n",
                "\n",
                "X_te_w = word_vec.transform(test_df['text'])\n",
                "X_te_c = char_vec.transform(test_df['text'])\n",
                "X_te_s = csr_matrix(extract_dense_stylometry(test_df['text']))\n",
                "X_test = hstack([X_te_w, X_te_c, X_te_s])\n",
                "print(f'Feature matrix: {X_train.shape[1]} features')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Fitting balanced classifier...')\n",
                "clf = LogisticRegression(C=2.5, max_iter=300, class_weight='balanced', solver='lbfgs', random_state=42)\n",
                "clf.fit(X_train, train_df['model'])\n",
                "\n",
                "preds = clf.predict(X_test)\n",
                "\n",
                "sub = pd.DataFrame({'id': test_df['id'], 'model': preds})\n",
                "\n",
                "# Rigorous validation\n",
                "assert len(sub) == len(test_df), 'Row count mismatch'\n",
                "assert not sub['model'].isnull().any(), 'Null values in predictions'\n",
                "assert set(sub['model']).issubset(set(OFFICIAL_18_CLASSES)), 'Illegal class predicted'\n",
                "\n",
                "sub.to_parquet('submission.parquet', index=False)\n",
                "print('SUCCESS! submission.parquet generated and validated!')\n",
                "print(sub.head())\n",
                "print(sub['model'].value_counts())\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/03_kaggle_submission_kernel.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Updated notebooks/03_kaggle_submission_kernel.ipynb successfully!")

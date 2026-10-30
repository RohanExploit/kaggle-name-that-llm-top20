import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score
from src.data import LABEL_TO_ID, create_prompt_grouped_folds

train_df = pd.read_csv("data/train.csv").dropna(subset=["text", "model", "prompt_id"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
train_df["label_id"] = train_df["model"].map(LABEL_TO_ID)

# Load neural OOF
neural_oof = np.load("checkpoints/gpu_transformer/oof_probs.npy")

# Load or compute stylometry OOF
from scripts.test_enhanced_stylometry import enhanced_stylometry, folds_df
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sty_oof_probs = np.zeros((len(folds_df), 18))
for fold in range(5):
    tr = folds_df[folds_df["fold"] != fold]
    va = folds_df[folds_df["fold"] == fold]
    
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)
    char_vec = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', min_df=3, max_features=10000, sublinear_tf=True)
    
    X_tr = hstack([word_vec.fit_transform(tr['text']), char_vec.fit_transform(tr['text']), csr_matrix(enhanced_stylometry(tr['text']))])
    X_va = hstack([word_vec.transform(va['text']), char_vec.transform(va['text']), csr_matrix(enhanced_stylometry(va['text']))])
    
    clf = LogisticRegression(C=2.5, max_iter=350, class_weight='balanced', solver='lbfgs', random_state=42)
    clf.fit(X_tr, tr['label_id'])
    sty_oof_probs[folds_df["fold"] == fold] = clf.predict_proba(X_va)

np.save("checkpoints/enhanced_sty_oof_probs.npy", sty_oof_probs)
print("Computed and saved enhanced_sty_oof_probs.npy!")

# Grid search optimal temperature and alpha
best_score = 0.0
best_params = None
targets = folds_df['label_id'].values

for alpha in np.linspace(0.2, 0.8, 13):
    for T_sty in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
        for T_neu in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
            # Temperature scale logits
            p_s = softmax(np.log(np.clip(sty_oof_probs, 1e-7, 1.0)) / T_sty, axis=1)
            p_n = softmax(np.log(np.clip(neural_oof, 1e-7, 1.0)) / T_neu, axis=1)
            
            blended = alpha * p_n + (1.0 - alpha) * p_s
            preds = np.argmax(blended, axis=1)
            score = balanced_accuracy_score(targets, preds)
            
            if score > best_score:
                best_score = score
                best_params = (alpha, T_neu, T_sty)

print(f"Optimal Ensemble: Alpha={best_params[0]:.2f} (Neural), T_neu={best_params[1]:.2f}, T_sty={best_params[2]:.2f} => Peak Balanced Acc: {best_score:.4f}")

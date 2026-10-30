import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# Load train dataset
df = pd.read_csv(r'E:\projects\kaggle\data\train.csv')
labels = sorted(df['model'].unique())
label2idx = {l: i for i, l in enumerate(labels)}
y_true = np.array([label2idx[m] for m in df['model']])

def apply_forensic_corrections(texts, prob_matrix, labels):
    """
    Applies empirical stylometric invariants:
    1. Claude and GLM never use curly quotes.
    2. Gemini almost never uses em-dashes.
    """
    p = np.array(prob_matrix, copy=True)
    claude_indices = [i for i, l in enumerate(labels) if l.startswith('claude')]
    glm_indices = [i for i, l in enumerate(labels) if l.startswith('glm')]
    gemini_indices = [i for i, l in enumerate(labels) if l.startswith('gemini')]
    
    for i, t in enumerate(texts):
        # Curly quotes check
        curly_count = t.count('“') + t.count('”') + t.count('‘') + t.count('’')
        if curly_count > 0:
            # Heavily penalize Claude and GLM
            p[i, claude_indices] *= 0.05
            p[i, glm_indices] *= 0.05
            
        # Em-dash check
        em_count = t.count('—')
        if em_count > 0:
            # Penalize Gemini
            p[i, gemini_indices] *= 0.10
            
    # Renormalize
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p

print("Forensic corrections module defined successfully.")

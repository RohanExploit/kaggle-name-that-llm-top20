"""
Test Hierarchical Family-Conditioned Classification
Hypothesis: Because 68.9% of samples have the correct family, applying family-level probability pooling
and intra-family calibration can resolve sibling variant ambiguities.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from scipy.special import softmax

OFFICIAL_18_CLASSES = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
LABEL_TO_ID = {name: i for i, name in enumerate(OFFICIAL_18_CLASSES)}

FAMILIES = {
    'OpenAI': [0, 1, 2, 3, 4],
    'Google': [5, 6, 7, 8],
    'Anthropic': [9, 10, 11, 12, 13],
    'Qwen': [14],
    'GLM': [15, 16],
    'Moonshot': [17]
}

def main():
    df = pd.read_csv('data/train.csv')
    df = df[df['model'].isin(OFFICIAL_18_CLASSES)].dropna(subset=['text', 'prompt_id']).reset_index(drop=True)
    y_true = df['model'].map(LABEL_TO_ID).values
    
    sty_oof = np.load('checkpoints/sty_oof_probs.npy')
    neural_oof = np.load('checkpoints/gpu_transformer/oof_probs.npy')[:len(df)]
    
    # Baseline 50/50 raw blend
    base_blend = 0.50 * neural_oof + 0.50 * sty_oof
    base_score = balanced_accuracy_score(y_true, np.argmax(base_blend, axis=1))
    print(f"Baseline 50/50 OOF Balanced Accuracy: {base_score:.4f}")
    
    # Calculate Family Probability for each sample
    # Neural model family confidence vs Stylometry family confidence
    neural_fam_probs = np.zeros((len(df), len(FAMILIES)))
    sty_fam_probs = np.zeros((len(df), len(FAMILIES)))
    
    for f_idx, (fam, class_indices) in enumerate(FAMILIES.items()):
        neural_fam_probs[:, f_idx] = np.sum(neural_oof[:, class_indices], axis=1)
        sty_fam_probs[:, f_idx] = np.sum(sty_oof[:, class_indices], axis=1)
        
    fam_blend = 0.60 * neural_fam_probs + 0.40 * sty_fam_probs
    
    # Family accuracy
    fam_true = []
    for y in y_true:
        for f_idx, (fam, class_indices) in enumerate(FAMILIES.items()):
            if y in class_indices:
                fam_true.append(f_idx)
                break
    fam_true = np.array(fam_true)
    fam_pred = np.argmax(fam_blend, axis=1)
    print(f"Blended Family Identification Accuracy: {np.mean(fam_true == fam_pred)*100:.2f}%")
    
    # Test Family Masking / Boosting:
    # If the family confidence is above threshold, scale down cross-family probabilities
    best_boost_score = base_score
    best_gamma = 1.0
    
    for gamma in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
        adjusted_probs = base_blend.copy()
        for i in range(len(df)):
            top_fam = fam_pred[i]
            for f_idx, (fam, class_indices) in enumerate(FAMILIES.items()):
                if f_idx == top_fam:
                    adjusted_probs[i, class_indices] *= gamma
                else:
                    adjusted_probs[i, class_indices] /= gamma
        
        score = balanced_accuracy_score(y_true, np.argmax(adjusted_probs, axis=1))
        if score > best_boost_score:
            best_boost_score = score
            best_gamma = gamma
            
    print(f"Optimal Family Sharpening Gamma: {best_gamma}")
    print(f"Sharpened OOF Balanced Accuracy: {best_boost_score:.4f} (Gain: {best_boost_score - base_score:+.4f})")

if __name__ == '__main__':
    main()

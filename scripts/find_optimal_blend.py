import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

OFFICIAL_18_CLASSES = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
LABEL_TO_ID = {name: i for i, name in enumerate(OFFICIAL_18_CLASSES)}

def main():
    df = pd.read_csv('data/train.csv')
    df = df[df['model'].isin(OFFICIAL_18_CLASSES)].dropna(subset=['text', 'prompt_id']).reset_index(drop=True)
    y_true = df['model'].map(LABEL_TO_ID).values
    
    sty_oof = np.load('checkpoints/sty_oof_probs.npy')
    neural_oof = np.load('checkpoints/gpu_transformer/oof_probs.npy')[:len(df)]
    
    best_score = 0.0
    best_params = None
    
    for alpha in [0.4, 0.45, 0.5, 0.55, 0.6]:
        for T_s in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]:
            for T_n in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]:
                p_s = softmax(np.log(np.clip(sty_oof, 1e-7, 1.0)) / T_s, axis=1)
                p_n = softmax(np.log(np.clip(neural_oof, 1e-7, 1.0)) / T_n, axis=1)
                blend = alpha * p_n + (1.0 - alpha) * p_s
                score = balanced_accuracy_score(y_true, np.argmax(blend, axis=1))
                if score > best_score:
                    best_score = score
                    best_params = (alpha, T_n, T_s)
                    
    print(f"Optimal Calibrated Ensemble:")
    print(f"  Alpha (Neural weight): {best_params[0]:.2f}")
    print(f"  Neural Temperature:   {best_params[1]:.2f}")
    print(f"  Stylometry Temperature: {best_params[2]:.2f}")
    print(f"  Peak Balanced Accuracy: {best_score:.4f} (Raw 50/50 was 0.3215)")

if __name__ == '__main__':
    main()

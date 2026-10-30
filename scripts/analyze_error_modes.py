"""
Analyze OOF Error Modes: Lab Family Confusion vs Sibling Model Confusion
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

OFFICIAL_18_CLASSES = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
LABEL_TO_ID = {name: i for i, name in enumerate(OFFICIAL_18_CLASSES)}

FAMILY_MAP = {
    'gpt-5.5': 'OpenAI', 'gpt-5.6-luna': 'OpenAI', 'gpt-5.6-sol': 'OpenAI', 'gpt-5.6-terra': 'OpenAI', 'gpt-6-astra': 'OpenAI',
    'gemini-3.1-pro-preview': 'Google', 'gemini-3.6-flash': 'Google', 'gemini-3.7-flash': 'Google', 'gemini-3.8-flash': 'Google',
    'claude-fable-5': 'Anthropic', 'claude-opus-4-6': 'Anthropic', 'claude-opus-4-8': 'Anthropic', 'claude-opus-5': 'Anthropic', 'claude-sonnet-5': 'Anthropic',
    'qwen3.8-max': 'Qwen', 'glm-5.3': 'GLM', 'glm-5.3-flash': 'GLM', 'kimi-k3': 'Moonshot'
}

def main():
    df = pd.read_csv('data/train.csv')
    df = df[df['model'].isin(OFFICIAL_18_CLASSES)].dropna(subset=['text', 'prompt_id']).reset_index(drop=True)
    y_true = df['model'].map(LABEL_TO_ID).values
    
    sty_oof = np.load('checkpoints/sty_oof_probs.npy')
    neural_oof = np.load('checkpoints/gpu_transformer/oof_probs.npy')[:len(df)]
    
    blend = 0.50 * neural_oof + 0.50 * sty_oof
    y_pred = np.argmax(blend, axis=1)
    
    score = balanced_accuracy_score(y_true, y_pred)
    print(f"50/50 Ensemble OOF Balanced Accuracy: {score:.4f}")
    
    # Calculate confusion breakdown
    true_families = [FAMILY_MAP[OFFICIAL_18_CLASSES[i]] for i in y_true]
    pred_families = [FAMILY_MAP[OFFICIAL_18_CLASSES[i]] for i in y_pred]
    
    correct_exact = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    correct_family = sum(1 for yt, yp in zip(true_families, pred_families) if yt == yp)
    total = len(y_true)
    
    print(f"Total validation samples: {total}")
    print(f"Exact Model Accuracy: {correct_exact}/{total} ({correct_exact/total*100:.1f}%)")
    print(f"Family-Level Accuracy: {correct_family}/{total} ({correct_family/total*100:.1f}%)")
    
    sibling_errors = correct_family - correct_exact
    cross_family_errors = total - correct_family
    print(f"\n--- Error Distribution ---")
    print(f"Sibling Errors (Right Family, Wrong Variant): {sibling_errors} ({sibling_errors/(total - correct_exact)*100:.1f}% of all errors)")
    print(f"Cross-Family Errors (Wrong Family): {cross_family_errors} ({cross_family_errors/(total - correct_exact)*100:.1f}% of all errors)")

if __name__ == '__main__':
    main()

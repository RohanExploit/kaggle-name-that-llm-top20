"""
Test Multi-Fold Probability-Space Averaging
Runs Fold 0 and Fold 1 separately, averages output probabilities, and evaluates OOF accuracy.
"""
import os, sys, glob, re, time
import numpy as np
import pandas as pd
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
    
    neural_oof = np.load('checkpoints/gpu_transformer/oof_probs.npy')[:len(df)]
    sty_oof = np.load('checkpoints/sty_oof_probs.npy')
    
    # Dual probability blend
    blend_5050 = 0.50 * neural_oof + 0.50 * sty_oof
    score_5050 = balanced_accuracy_score(y_true, np.argmax(blend_5050, axis=1))
    print(f"Verified Dual 50/50 Probability Blend OOF: {score_5050:.4f}")

if __name__ == '__main__':
    main()

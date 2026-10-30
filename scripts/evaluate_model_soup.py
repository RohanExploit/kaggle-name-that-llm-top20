import os, sys, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import balanced_accuracy_score
from safetensors.torch import save_file

OFFICIAL_18_CLASSES = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
LABEL_TO_ID = {name: i for i, name in enumerate(OFFICIAL_18_CLASSES)}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print(f"Creating 3-Fold Model Soup on device: {DEVICE}")
    f0 = torch.load('checkpoints/gpu_transformer/deberta_fold_0.pt', map_location='cpu', weights_only=True)
    f1 = torch.load('checkpoints/gpu_transformer/deberta_fold_1.pt', map_location='cpu', weights_only=True)
    f2 = torch.load('checkpoints/gpu_transformer/deberta_fold_2.pt', map_location='cpu', weights_only=True)
    
    soup_state_dict = {}
    for k in f0.keys():
        soup_state_dict[k] = (f0[k].float() + f1[k].float() + f2[k].float()) / 3.0
        
    print(f"Averaged {len(soup_state_dict)} tensor parameters successfully.")
    
    # Save soup safetensors and pt
    os.makedirs('checkpoints/soup_model', exist_ok=True)
    save_file(soup_state_dict, 'checkpoints/soup_model/model.safetensors')
    print("Saved model soup to checkpoints/soup_model/model.safetensors")
    
    # Copy config and tokenizer files
    for fn in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'dataset-metadata.json']:
        src = os.path.join('checkpoints/high_score_model', fn)
        dst = os.path.join('checkpoints/soup_model', fn)
        if os.path.exists(src):
            with open(src, 'rb') as f_in, open(dst, 'wb') as f_out:
                f_out.write(f_in.read())
    print("Copied tokenizer and configuration metadata.")
    
if __name__ == '__main__':
    main()

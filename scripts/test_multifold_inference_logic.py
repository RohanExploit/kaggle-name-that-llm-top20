"""
Test Multi-Fold Sequential Inference Logic Locally
"""
import os, glob, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import load_file

OFFICIAL_18_CLASSES = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    model_dir = 'checkpoints/high_score_model'
    test_df = pd.read_csv('data/test.csv')
    print(f"Testing on device {DEVICE} with {len(test_df)} test rows")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    class NTLTestDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=256):
            self.texts = list(texts)
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tokenizer(str(self.texts[idx]), max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0)}
            
    test_loader = DataLoader(NTLTestDataset(test_df['text'], tokenizer, 256), batch_size=8, shuffle=False)
    
    fold_files = [
        os.path.join(model_dir, 'model.safetensors'),
        os.path.join(model_dir, 'model_fold0.safetensors')
    ]
    
    all_fold_probs = []
    for f_idx, f_path in enumerate(fold_files):
        print(f"Loading {f_path}...")
        sd = load_file(f_path)
        config = AutoConfig.from_pretrained(model_dir, num_labels=18)
        model = AutoModelForSequenceClassification.from_config(config).to(DEVICE)
        model.load_state_dict(sd)
        model.eval()
        
        fold_p = []
        with torch.no_grad():
            for batch in test_loader:
                ids, mask = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE)
                out = model(ids, attention_mask=mask)
                probs = torch.softmax(out.logits, dim=-1)
                fold_p.extend(probs.cpu().numpy())
                
        all_fold_probs.append(np.array(fold_p))
        del model, sd
        torch.cuda.empty_cache()
        
    avg_probs = np.mean(all_fold_probs, axis=0)
    print(f"SUCCESS! Multi-fold probability average shape: {avg_probs.shape}")
    print(f"Sample predictions: {[OFFICIAL_18_CLASSES[i] for i in np.argmax(avg_probs, axis=1)]}")

if __name__ == '__main__':
    main()

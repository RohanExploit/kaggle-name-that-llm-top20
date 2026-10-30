"""
Prepare Multi-Fold Neural Checkpoint Bundle for Probability-Space Ensembling
Converts Fold 0 and Fold 1 PyTorch state dicts to safetensors in separate subfolders.
"""
import os, sys, shutil
import torch
from safetensors.torch import save_file

def main():
    print("Preparing multi-fold neural bundle...")
    os.makedirs('checkpoints/multi_fold_bundle/fold0', exist_ok=True)
    os.makedirs('checkpoints/multi_fold_bundle/fold1', exist_ok=True)
    
    # 1. Convert Fold 0
    f0_path = 'checkpoints/gpu_transformer/deberta_fold_0.pt'
    print(f"Loading {f0_path}...")
    sd0 = torch.load(f0_path, map_location='cpu', weights_only=True)
    save_file(sd0, 'checkpoints/multi_fold_bundle/fold0/model.safetensors')
    print("Saved fold0/model.safetensors")
    del sd0
    
    # 2. Convert Fold 1
    f1_path = 'checkpoints/gpu_transformer/deberta_fold_1.pt'
    print(f"Loading {f1_path}...")
    sd1 = torch.load(f1_path, map_location='cpu', weights_only=True)
    save_file(sd1, 'checkpoints/multi_fold_bundle/fold1/model.safetensors')
    print("Saved fold1/model.safetensors")
    del sd1
    
    # 3. Copy config & tokenizer files
    for fn in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        src = os.path.join('checkpoints/high_score_model', fn)
        dst = os.path.join('checkpoints/multi_fold_bundle', fn)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            shutil.copyfile(src, os.path.join('checkpoints/multi_fold_bundle/fold0', fn))
            shutil.copyfile(src, os.path.join('checkpoints/multi_fold_bundle/fold1', fn))
            
    print("SUCCESS! Multi-fold bundle prepared successfully in checkpoints/multi_fold_bundle/")

if __name__ == '__main__':
    main()

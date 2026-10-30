"""
Autonomous Local GPU Worker (NVIDIA RTX 2050 - 4GB VRAM).
Trains PyTorch Transformer (DeBERTa-v3) with FP16 and gradient accumulation,
generates neural out-of-fold predictions, blends with stylometric baseline,
and pushes submissions directly to the Kaggle Leaderboard.
"""

import gc
import os
import subprocess
import sys
import time

# Ensure project root is in sys.path and fix HuggingFace cache directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache", "huggingface")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, "cache", "torch")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import balanced_accuracy_score

from src.data import (
    OFFICIAL_18_CLASSES,
    LABEL_TO_ID,
    ID_TO_LABEL,
    PROVIDER_MAP,
    create_prompt_grouped_folds,
)
from src.metrics import evaluate_predictions, print_evaluation_summary


MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LEN = 256
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
LR = 2.5e-5
EPOCHS = 3
CHECKPOINT_DIR = "checkpoints/gpu_transformer"


class PassageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def run_gpu_cycle():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65, flush=True)
    print(f"[GPU ENGINE] Initializing on {device}: {torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 65, flush=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data: official train.csv + balanced sample from master pool
    official_train = pd.read_csv("data/train.csv")
    official_train["source"] = "official_train"
    print(f"Loaded {len(official_train):,} official ground-truth passages.", flush=True)

    master_path = "data/master_train_pool.parquet"
    if os.path.exists(master_path):
        master_df = pd.read_parquet(master_path)
        # Sample 5,000 extra passages to enrich representation
        extra_sample = master_df.groupby("model", group_keys=False).apply(
            lambda x: x.sample(min(len(x), 300), random_state=42)
        ).reset_index(drop=True)
        combined_df = pd.concat([official_train, extra_sample], ignore_index=True)
    else:
        combined_df = official_train

    combined_df = combined_df.dropna(subset=["text", "model", "prompt_id"]).drop_duplicates(subset=["text"])
    combined_df = combined_df[combined_df["model"].isin(OFFICIAL_18_CLASSES)].reset_index(drop=True)
    combined_df["label_id"] = combined_df["model"].map(LABEL_TO_ID)

    print(f"[GPU] Total training corpus: {len(combined_df):,} samples across 18 models", flush=True)

    # 2. Split with strict prompt-grouping
    df = create_prompt_grouped_folds(combined_df, n_splits=5, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Train across folds
    oof_probs = np.zeros((len(df), len(OFFICIAL_18_CLASSES)), dtype=np.float32)

    for fold in range(3):  # Train top 3 folds for high-efficiency overnight ensemble
        t0 = time.time()
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)

        train_loader = DataLoader(
            PassageDataset(train_df["text"], train_df["label_id"], tokenizer, MAX_LEN),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_loader = DataLoader(
            PassageDataset(val_df["text"], val_df["label_id"], tokenizer, MAX_LEN),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=18).to(device)

        # Class balanced weights
        class_counts = train_df["label_id"].value_counts().sort_index().values
        weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32).to(device)
        weights = (weights / weights.sum()) * 18.0
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
        save_path = os.path.join(CHECKPOINT_DIR, f"deberta_fold_{fold}.pt")
        val_probs = []

        if os.path.exists(save_path):
            print(f"\n[GPU FOLD {fold+1}/3] Found existing checkpoint at {save_path}. Loading weights for validation...", flush=True)
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    with torch.cuda.amp.autocast():
                        out = model(ids, attention_mask=mask)
                        probs = torch.softmax(out.logits, dim=-1)
                    preds.extend(torch.argmax(probs, dim=-1).cpu().numpy())
                    targets.extend(labels.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
            bal_acc = balanced_accuracy_score(np.array(targets), np.array(preds))
            print(f"[GPU FOLD {fold+1}/3] Loaded checkpoint! Val Balanced Acc: {bal_acc:.4f}", flush=True)
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
            )
            scaler = torch.cuda.amp.GradScaler(enabled=True)

            print(f"\n[GPU FOLD {fold+1}/3] Training {EPOCHS} epochs...", flush=True)

            for epoch in range(EPOCHS):
                model.train()
                optimizer.zero_grad()
                total_loss = 0.0

                for step, batch in enumerate(train_loader):
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    with torch.cuda.amp.autocast():
                        out = model(ids, attention_mask=mask)
                        loss = criterion(out.logits, labels) / GRAD_ACCUM_STEPS

                    scaler.scale(loss).backward()

                    if (step + 1) % GRAD_ACCUM_STEPS == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()

                    total_loss += loss.item() * GRAD_ACCUM_STEPS
                    if (step + 1) % 100 == 0:
                        print(f"    [Fold {fold+1}] Epoch {epoch+1} step {step+1}/{len(train_loader)} | loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}", flush=True)

                # Val evaluation
                model.eval()
                preds, targets, val_probs = [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        ids = batch["input_ids"].to(device)
                        mask = batch["attention_mask"].to(device)
                        labels = batch["label"].to(device)
                        with torch.cuda.amp.autocast():
                            out = model(ids, attention_mask=mask)
                            probs = torch.softmax(out.logits, dim=-1)
                        preds.extend(torch.argmax(probs, dim=-1).cpu().numpy())
                        targets.extend(labels.cpu().numpy())
                        val_probs.extend(probs.cpu().numpy())
                bal_acc = balanced_accuracy_score(np.array(targets), np.array(preds))
                print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Balanced Acc: {bal_acc:.4f}", flush=True)

                del preds, targets
                gc.collect()
                torch.cuda.empty_cache()

            # Save model for this fold
            torch.save(model.state_dict(), save_path)
            print(f"Saved fold {fold+1} weights to {save_path}", flush=True)

        oof_probs[df["fold"] == fold] = np.array(val_probs)

        # Free fold resources from GPU memory
        del model, val_probs
        if "optimizer" in locals():
            del optimizer
        if "scheduler" in locals():
            del scheduler
        gc.collect()
        torch.cuda.empty_cache()

    # Save overall OOF predictions and print overall cross-validation score
    np.save(os.path.join(CHECKPOINT_DIR, "oof_probs.npy"), oof_probs)
    valid_mask = oof_probs.sum(axis=1) > 0
    if valid_mask.sum() > 0:
        overall_bal_acc = balanced_accuracy_score(df["label_id"][valid_mask], np.argmax(oof_probs[valid_mask], axis=1))
        print(f"\n[GPU ENGINE] Overall Neural 3-Fold Balanced Accuracy: {overall_bal_acc:.4f}", flush=True)

    # 3. Generate Submission on Official test.csv
    test_df = pd.read_csv("data/test.csv")
    test_loader = DataLoader(
        PassageDataset(test_df["text"], None, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    ensemble_test_probs = np.zeros((len(test_df), 18), dtype=np.float32)

    for fold in range(3):
        weights_path = os.path.join(CHECKPOINT_DIR, f"deberta_fold_{fold}.pt")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=18).to(device)
        model.load_state_dict(torch.load(weights_path))
        model.eval()

        fold_test_probs = []
        with torch.no_grad():
            for batch in test_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                with torch.cuda.amp.autocast():
                    out = model(ids, attention_mask=mask)
                    probs = torch.softmax(out.logits, dim=-1)
                fold_test_probs.extend(probs.cpu().numpy())
        ensemble_test_probs += np.array(fold_test_probs)

    ensemble_test_probs /= 3.0
    pred_indices = np.argmax(ensemble_test_probs, axis=1)
    predicted_models = [OFFICIAL_18_CLASSES[i] for i in pred_indices]

    sub = pd.DataFrame({"id": test_df["id"], "model": predicted_models})
    sub.to_parquet("submission.parquet", index=False)
    sub.to_csv("submission.csv", index=False)
    print(f"\n[GPU ENGINE] Updated submission.parquet with DeBERTa-v3 Neural Ensemble ({len(sub)} rows)!", flush=True)

    # 4. Push directly to Kaggle Leaderboard!
    try:
        print("[KAGGLE API] Submitting submission.parquet directly to Kaggle Leaderboard...", flush=True)
        res = subprocess.run(
            ["kaggle", "competitions", "submit", "-c", "name-that-llm", "-f", "submission.parquet", "-m", "DeBERTa-v3 FP16 Neural Ensemble"],
            capture_output=True,
            text=True,
        )
        print(res.stdout, flush=True)
        if res.stderr:
            print(res.stderr, flush=True)
    except Exception as e:
        print(f"Kaggle submit notification: {e}", flush=True)


if __name__ == "__main__":
    run_gpu_cycle()

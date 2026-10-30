"""
Production Neural Transformer Fine-Tuning Script for Name That LLM.
Optimized for Colab and Kaggle GPU clusters (T4, P100, L4, A100).
Trains DeBERTa-v3 / ModernBERT on the expanded 197k-passage corpus.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import balanced_accuracy_score
from src.data import OFFICIAL_18_CLASSES, LABEL_TO_ID, create_prompt_grouped_folds
from src.metrics import evaluate_predictions, print_evaluation_summary


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=384):
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


def train_transformer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")
    if device.type == "cuda":
        print(f"GPU Accelerator: {torch.cuda.get_device_name(0)}")

    # 1. Load data
    print(f"Loading master dataset from: {args.data_path}")
    if args.data_path.endswith(".parquet"):
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path)

    # Subsample if requested for faster iteration
    if args.sample_size and len(df) > args.sample_size:
        print(f"Subsampling to {args.sample_size:,} balanced samples for accelerated training...")
        df = df.groupby("model", group_keys=False).apply(
            lambda x: x.sample(min(len(x), args.sample_size // 18), random_state=args.seed)
        ).reset_index(drop=True)

    df["label_id"] = df["model"].map(LABEL_TO_ID)
    print(f"Active training corpus: {len(df):,} samples across 18 models")

    # 2. Split with strict prompt-grouping
    df = create_prompt_grouped_folds(df, n_splits=args.n_splits, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Train fold 0 (or all folds if requested)
    train_mask = df["fold"] != args.fold
    val_mask = df["fold"] == args.fold

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    print(f"Fold {args.fold} - Train: {len(train_df):,}, Val: {len(val_df):,}")

    train_loader = DataLoader(
        TextDataset(train_df["text"], train_df["label_id"], tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        TextDataset(val_df["text"], val_df["label_id"], tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=18
    ).to(device)

    # Compute inverse class frequencies for balanced cross-entropy
    class_counts = train_df["label_id"].value_counts().sort_index().values
    class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * 18.0
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_score = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(ids, attention_mask=mask)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Step {step+1}/{len(train_loader)} | Loss: {total_loss / (step + 1):.4f}", flush=True)

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    out = model(ids, attention_mask=mask)
                preds.extend(torch.argmax(out.logits, dim=-1).cpu().numpy())
                targets.extend(labels.cpu().numpy())

        eval_res = evaluate_predictions(np.array(targets), np.array(preds), OFFICIAL_18_CLASSES)
        bal_acc = eval_res["balanced_accuracy"]
        print(f"\n--- Epoch {epoch+1} Validation Balanced Accuracy: {bal_acc:.4f} ---")

        if bal_acc > best_val_score:
            best_val_score = bal_acc
            save_path = os.path.join(args.output_dir, f"transformer_best_fold_{args.fold}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"⭐ New Best Model Checkpoint Saved to {save_path} (Balanced Acc: {best_val_score:.4f})")

    print(f"\nTraining Complete. Peak Balanced Accuracy: {best_val_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/master_train_pool.parquet")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-small")
    parser.add_argument("--sample_size", type=int, default=20000, help="Subsample for speed, 0 for all")
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints/transformer")
    args = parser.parse_args()

    train_transformer(args)

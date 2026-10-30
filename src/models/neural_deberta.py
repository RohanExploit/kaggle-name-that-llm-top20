"""
PyTorch & HuggingFace Transformer Classifier for Name That LLM.
Optimized for Google Colab and Kaggle GPUs (T4, P100, L4, A100).
Supports DeBERTa-v3, RoBERTa, ModernBERT, and Qwen backbones.
"""

import os
from typing import Dict, List, Optional, Tuple
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
from src.data import OFFICIAL_18_CLASSES, LABEL_TO_ID, PROVIDER_MAP


class PassageDataset(Dataset):
    """PyTorch Dataset for text passages."""

    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_length: int = 384):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BalancedCrossEntropyLoss(nn.Module):
    """Computes cross-entropy loss with class weights inversely proportional to class frequencies."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Trains for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluates validation loader and returns balanced accuracy, predictions, and probabilities."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        preds = torch.argmax(probs, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    probs = np.array(all_probs)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    return bal_acc, y_pred, probs

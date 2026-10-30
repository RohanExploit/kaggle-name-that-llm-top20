"""
Data loading, validation, and prompt-grouped splitting for Name That LLM.
Enforces strict GroupKFold on prompt_id to prevent semantic topic leakage.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold


OFFICIAL_18_CLASSES = [
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
]

LABEL_TO_ID = {name: idx for idx, name in enumerate(OFFICIAL_18_CLASSES)}
ID_TO_LABEL = {idx: name for idx, name in enumerate(OFFICIAL_18_CLASSES)}

PROVIDER_MAP = {
    # OpenAI
    "gpt-5.5": "openai",
    "gpt-5.6-luna": "openai",
    "gpt-5.6-sol": "openai",
    "gpt-5.6-terra": "openai",
    "gpt-6-astra": "openai",
    # Google
    "gemini-3.1-pro-preview": "google",
    "gemini-3.6-flash": "google",
    "gemini-3.7-flash": "google",
    "gemini-3.8-flash": "google",
    # Anthropic
    "claude-fable-5": "anthropic",
    "claude-opus-4-6": "anthropic",
    "claude-opus-4-8": "anthropic",
    "claude-opus-5": "anthropic",
    "claude-sonnet-5": "anthropic",
    # Qwen
    "qwen3.8-max": "qwen",
    # ZAI
    "glm-5.3": "zai",
    "glm-5.3-flash": "zai",
    # Moonshot
    "kimi-k3": "moonshot",
}


def load_dataset(data_path: str = "data/ref_dataset/passages_clean.csv") -> pd.DataFrame:
    """
    Loads dataset, standardizes columns to [id, prompt_id, model, text],
    filters to valid official 18 models, and removes empty or invalid passages.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    df = pd.read_csv(data_path)

    # Standardize column names
    col_mapping = {}
    if "label" in df.columns and "model" not in df.columns:
        col_mapping["label"] = "model"
    if "passage" in df.columns and "text" not in df.columns:
        col_mapping["passage"] = "text"
    if col_mapping:
        df = df.rename(columns=col_mapping)

    required_cols = ["id", "prompt_id", "model", "text"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df[required_cols].copy()

    # Filter out missing or malformed text
    df = df.dropna(subset=["text", "model", "prompt_id"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]

    # Filter to official 18 classes
    df = df[df["model"].isin(OFFICIAL_18_CLASSES)].copy()
    df["label_id"] = df["model"].map(LABEL_TO_ID)
    df["provider"] = df["model"].map(PROVIDER_MAP)

    df = df.reset_index(drop=True)
    return df


def create_prompt_grouped_folds(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
    """
    Splits DataFrame into n_splits folds grouped by prompt_id.
    Uses StratifiedGroupKFold where possible to balance class distributions across folds.
    Guarantees no prompt_id appears in both train and validation of any fold.
    """
    df = df.copy()
    df["fold"] = -1

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # In case StratifiedGroupKFold struggles with singletons, fallback to GroupKFold
    try:
        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(df, y=df["label_id"], groups=df["prompt_id"])
        ):
            df.loc[val_idx, "fold"] = fold
    except Exception:
        gkf = GroupKFold(n_splits=n_splits)
        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(df, y=df["label_id"], groups=df["prompt_id"])
        ):
            df.loc[val_idx, "fold"] = fold

    # Validate leakage
    for f in range(n_splits):
        train_prompts = set(df[df["fold"] != f]["prompt_id"])
        val_prompts = set(df[df["fold"] == f]["prompt_id"])
        overlap = train_prompts.intersection(val_prompts)
        assert len(overlap) == 0, f"LEAKAGE DETECTED in Fold {f}: {len(overlap)} overlapping prompts!"

    return df

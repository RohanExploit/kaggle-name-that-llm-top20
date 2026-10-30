"""
Dataset Expansion and Normalization Engine for Name That LLM.
Aggregates multiple verified external generation datasets into a single,
clean, deduplicated master training pool strictly mapped to the 18 target models.
"""

import glob
import os
from typing import List, Optional
import numpy as np
import pandas as pd

from src.data import OFFICIAL_18_CLASSES, PROVIDER_MAP, LABEL_TO_ID


OUTPUT_MASTER_PATH = "data/master_train_pool.parquet"


def normalize_record(df: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    """Normalizes arbitrary dataframe to standard schema: [id, prompt_id, model, text, source]."""
    cols = {c.lower(): c for c in df.columns}

    # Label / model column
    model_col = None
    for cand in ["model", "label", "target"]:
        if cand in cols:
            model_col = cols[cand]
            break

    # Text / passage column
    text_col = None
    for cand in ["text", "passage", "generation", "content"]:
        if cand in cols:
            text_col = cols[cand]
            break

    # Prompt ID column
    prompt_col = cols.get("prompt_id", cols.get("prompt", None))

    # ID column
    id_col = cols.get("id", None)

    if not model_col or not text_col:
        print(f"Skipping incompatible dataset ({source_tag})")
        return pd.DataFrame()

    out = pd.DataFrame()
    out["model"] = df[model_col].astype(str).str.strip()
    out["text"] = df[text_col].astype(str).str.strip()

    if id_col:
        out["id"] = df[id_col].astype(str)
    else:
        out["id"] = [f"{source_tag}_{i}" for i in range(len(df))]

    if prompt_col:
        out["prompt_id"] = df[prompt_col].astype(str)
    else:
        out["prompt_id"] = [f"{source_tag}_p{i // 18}" for i in range(len(df))]

    out["source"] = source_tag
    return out


def build_master_dataset():
    """Finds all downloaded datasets and merges them into a clean, deduplicated pool."""
    print("=" * 65)
    print("  BUILDING EXPANDED MASTER TRAINING DATASET")
    print("=" * 65)

    all_dfs = []

    # 1. Clean Reference Dataset
    ref_path = "data/ref_dataset/passages_clean.csv"
    if os.path.exists(ref_path):
        print(f"Reading: {ref_path}")
        df_ref = pd.read_csv(ref_path)
        norm_ref = normalize_record(df_ref, "ref_clean")
        all_dfs.append(norm_ref)

    # 2. Extra Claude API Passages
    claude_path = "data/extra_passages/ntl_extra_passages_claude.csv"
    if os.path.exists(claude_path):
        print(f"Reading: {claude_path}")
        df_cl = pd.read_csv(claude_path)
        norm_cl = normalize_record(df_cl, "extra_claude")
        all_dfs.append(norm_cl)

    # 3. Extra GPT & Gemini Generations
    gpt_path = "data/extra_gpt_gemini/ntl_generations.csv"
    if os.path.exists(gpt_path):
        print(f"Reading: {gpt_path}")
        df_gpt = pd.read_csv(gpt_path)
        norm_gpt = normalize_record(df_gpt, "extra_gpt_gemini")
        all_dfs.append(norm_gpt)

    # 4. Open Pool Parquet (193k Clean Passages)
    open_pool_path = "data/open_pool/ntl_open_pool.parquet"
    if os.path.exists(open_pool_path):
        print(f"Reading: {open_pool_path}")
        df_open = pd.read_parquet(open_pool_path)
        norm_open = normalize_record(df_open, "open_pool_v9")
        all_dfs.append(norm_open)

    if not all_dfs:
        raise FileNotFoundError("No datasets found to merge!")

    master = pd.concat(all_dfs, ignore_index=True)
    print(f"\nRaw aggregated rows: {len(master):,}")

    # Clean and filter
    # Filter strictly to official 18 target models
    master = master[master["model"].isin(OFFICIAL_18_CLASSES)].copy()
    print(f"After filtering to official 18 models: {len(master):,}")

    # Remove empty, too short, or corrupted text
    master = master.dropna(subset=["text", "model"])
    master = master[master["text"].str.len() >= 30]

    # Deduplicate exact duplicate text passages
    master = master.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"After deduplication: {len(master):,} unique passages")

    # Add provider mapping and label IDs
    master["provider"] = master["model"].map(PROVIDER_MAP)
    master["label_id"] = master["model"].map(LABEL_TO_ID)

    # Save to high-performance parquet
    os.makedirs(os.path.dirname(OUTPUT_MASTER_PATH), exist_ok=True)
    master.to_parquet(OUTPUT_MASTER_PATH, index=False)
    print(f"\nMaster training dataset successfully written to {OUTPUT_MASTER_PATH} ({os.path.getsize(OUTPUT_MASTER_PATH) / 1024 / 1024:.2f} MB)")

    print("\n--- Model Class Distribution in Master Pool ---")
    for model_name in OFFICIAL_18_CLASSES:
        count = (master["model"] == model_name).sum()
        print(f"  - {model_name:<28}: {count:,}")

    print("\n--- Provider Distribution ---")
    for prov, count in master["provider"].value_counts().items():
        print(f"  - {prov:<12}: {count:,}")

    return master


if __name__ == "__main__":
    build_master_dataset()

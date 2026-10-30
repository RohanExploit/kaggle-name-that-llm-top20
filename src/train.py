"""
Main training pipeline for Name That LLM.
Executes prompt-grouped cross validation and saves checkpoints.
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd

from src.data import (
    load_dataset,
    create_prompt_grouped_folds,
    OFFICIAL_18_CLASSES,
    PROVIDER_MAP,
)
from src.models.baseline_lgbm import BaselineModel
from src.metrics import evaluate_predictions, print_evaluation_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train Name That LLM Models")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/ref_dataset/passages_clean.csv",
        help="Path to training passages CSV",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of GroupKFold splits")
    parser.add_argument("--c_param", type=float, default=2.5, help="Regularization C parameter")
    parser.add_argument("--output_dir", type=str, default="checkpoints/baseline", help="Checkpoint output dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("  KAGGLE: NAME THAT LLM - TRAINING PIPELINE")
    print(f"  Target Classes: {len(OFFICIAL_18_CLASSES)}")
    print(f"  Dataset:        {args.data_path}")
    print(f"  Group Splits:   {args.n_splits}")
    print("=" * 65)

    # 1. Load Data
    print("\n[1/3] Loading dataset...")
    df = load_dataset(args.data_path)
    print(f"Loaded {len(df):,} valid samples across {df['prompt_id'].nunique():,} unique prompts.")
    print("Samples per provider:")
    for prov, count in df["provider"].value_counts().items():
        print(f"  - {prov:<12}: {count:,}")

    # 2. Form strict prompt-grouped folds
    print("\n[2/3] Generating strict prompt-grouped folds (preventing topic leakage)...")
    df = create_prompt_grouped_folds(df, n_splits=args.n_splits, seed=args.seed)

    # 3. Train Cross-Validated Model
    print("\n[3/3] Training model with Balanced Accuracy optimization...")
    model = BaselineModel(c_param=args.c_param)
    oof_probs, eval_results = model.fit_cv(
        df, n_splits=args.n_splits, output_dir=args.output_dir
    )

    # Save summary report
    os.makedirs("outputs", exist_ok=True)
    report_path = os.path.join("outputs", "train_summary.yaml")
    with open(report_path, "w") as f:
        yaml.dump(
            {
                "balanced_accuracy": float(eval_results["balanced_accuracy"]),
                "accuracy": float(eval_results["accuracy"]),
                "provider_accuracy": float(eval_results.get("provider_accuracy", 0.0)),
                "sibling_error_fraction": float(eval_results.get("sibling_error_fraction", 0.0)),
                "n_samples": len(df),
                "n_prompts": int(df["prompt_id"].nunique()),
            },
            f,
        )
    print(f"\nSummary metrics written to {report_path}")
    print("Training finished successfully.")


if __name__ == "__main__":
    main()

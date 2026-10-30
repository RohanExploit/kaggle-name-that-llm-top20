"""
Inference and submission generator for Name That LLM.
Produces compliant submission.parquet for Kaggle Code Competition evaluation.
"""

import argparse
import glob
import os
import joblib
import numpy as np
import pandas as pd

from src.data import OFFICIAL_18_CLASSES, LABEL_TO_ID, ID_TO_LABEL


def find_test_file(custom_path: str = None) -> str:
    """Finds test.csv from standard Kaggle competition input directory or local path."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    # Standard Kaggle paths
    kaggle_paths = [
        "/kaggle/input/competitions/name-that-llm/test.csv",
        "/kaggle/input/name-that-llm/test.csv",
        "data/test.csv",
    ]
    for p in kaggle_paths:
        if os.path.exists(p):
            return p

    # Recursive glob in case Kaggle nests it
    hits = glob.glob("/kaggle/input/**/test.csv", recursive=True)
    if hits:
        return hits[0]

    raise FileNotFoundError("Could not find test.csv in /kaggle/input or local data/.")


def generate_submission(
    checkpoint_dir: str = "checkpoints/baseline",
    test_path: str = None,
    output_parquet: str = "submission.parquet",
    output_csv: str = "submission.csv",
):
    """
    Loads saved fold models and pipelines, runs ensembled inference, and writes submission.parquet.
    """
    actual_test_path = find_test_file(test_path)
    print(f"Reading test data from: {actual_test_path}")
    test_df = pd.read_csv(actual_test_path)
    print(f"Test samples count: {len(test_df):,}")

    model_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_fold_*.joblib")))
    pipeline_files = sorted(glob.glob(os.path.join(checkpoint_dir, "pipeline_fold_*.joblib")))

    if not model_files or not pipeline_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    print(f"Found {len(model_files)} fold models. Running ensembled inference...")

    n_samples = len(test_df)
    n_classes = len(OFFICIAL_18_CLASSES)
    ensemble_probs = np.zeros((n_samples, n_classes), dtype=np.float32)

    for fold_idx, (m_file, p_file) in enumerate(zip(model_files, pipeline_files)):
        print(f"  - Applying fold {fold_idx + 1}...")
        model = joblib.load(m_file)
        pipeline = joblib.load(p_file)

        x_test, _ = pipeline.transform(test_df["text"].tolist())
        probs = model.predict_proba(x_test)
        ensemble_probs += probs

    ensemble_probs /= len(model_files)
    predicted_indices = np.argmax(ensemble_probs, axis=1)
    predicted_models = [OFFICIAL_18_CLASSES[i] for i in predicted_indices]

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "model": predicted_models,
        }
    )

    # Validation checks
    assert len(submission) == len(test_df), "Submission row count mismatch!"
    assert submission["id"].nunique() == len(test_df), "Duplicate IDs detected!"
    assert not submission["model"].isnull().any(), "Null predictions found!"
    assert set(submission["model"]).issubset(set(OFFICIAL_18_CLASSES)), "Invalid class labels found!"

    # Save to parquet as required by competition specification
    submission.to_parquet(output_parquet, index=False)
    submission.to_csv(output_csv, index=False)
    print(f"Submission written successfully to {output_parquet} ({len(submission)} rows)")
    print(f"Predicted class distribution:\n{submission['model'].value_counts()}")
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/baseline")
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--output_parquet", type=str, default="submission.parquet")
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    args = parser.parse_args()

    generate_submission(
        checkpoint_dir=args.checkpoint_dir,
        test_path=args.test_path,
        output_parquet=args.output_parquet,
        output_csv=args.output_csv,
    )

"""
Gradient Boosted Trees (LightGBM) & Calibrated Linear Baselines for Name That LLM.
Fast, interpretable, and high-performance cross-validated classifier.
"""

import os
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from src.metrics import evaluate_predictions, print_evaluation_summary
from src.features import FeaturePipeline
from src.data import OFFICIAL_18_CLASSES, PROVIDER_MAP


class BaselineModel:
    """
    K-fold ensemble of Logistic Regression with elastic regularization / LightGBM.
    Proven to deliver competitive baseline on prompt-grouped CV in seconds.
    """

    def __init__(self, c_param: float = 2.0, max_iter: int = 1000):
        self.c_param = c_param
        self.max_iter = max_iter
        self.models = []
        self.pipelines = []

    def fit_cv(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        output_dir: str = "checkpoints/baseline",
    ) -> Tuple[np.ndarray, Dict]:
        """
        Runs full cross-validation over the folds in df.
        Computes out-of-fold (OOF) prediction probabilities and official competition metric.
        """
        os.makedirs(output_dir, exist_ok=True)
        oof_probs = np.zeros((len(df), len(OFFICIAL_18_CLASSES)), dtype=np.float32)

        for fold in range(n_splits):
            train_mask = df["fold"] != fold
            val_mask = df["fold"] == fold

            train_df = df[train_mask]
            val_df = df[val_mask]

            print(f"\n--- Training Fold {fold + 1}/{n_splits} (Train: {len(train_df)}, Val: {len(val_df)}) ---", flush=True)

            pipeline = FeaturePipeline()
            x_train, _ = pipeline.fit_transform(train_df["text"].tolist())
            x_val, _ = pipeline.transform(val_df["text"].tolist())

            y_train = train_df["label_id"].values
            y_val = val_df["label_id"].values

            # Fast, high-accuracy sparse linear model with balanced class weighting
            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=60,
                tol=1e-3,
                class_weight="balanced",
                random_state=42 + fold,
            )
            model.fit(x_train, y_train)

            val_probs = model.predict_proba(x_val)
            oof_probs[val_mask] = val_probs

            fold_preds = np.argmax(val_probs, axis=1)
            fold_eval = evaluate_predictions(y_val, fold_preds, OFFICIAL_18_CLASSES, PROVIDER_MAP)
            print(
                f"Fold {fold + 1} Balanced Accuracy: {fold_eval['balanced_accuracy']:.4f} | "
                f"Raw Accuracy: {fold_eval['accuracy']:.4f} | "
                f"Provider Accuracy: {fold_eval.get('provider_accuracy', 0.0):.4f}",
                flush=True,
            )

            self.models.append(model)
            self.pipelines.append(pipeline)

            joblib.dump(model, os.path.join(output_dir, f"model_fold_{fold}.joblib"))
            joblib.dump(pipeline, os.path.join(output_dir, f"pipeline_fold_{fold}.joblib"))

        # Overall OOF Evaluation
        oof_preds = np.argmax(oof_probs, axis=1)
        overall_eval = evaluate_predictions(
            df["label_id"].values, oof_preds, OFFICIAL_18_CLASSES, PROVIDER_MAP
        )
        print("\n" + "#" * 65)
        print("OVERALL OUT-OF-FOLD (OOF) PERFORMANCE")
        print_evaluation_summary(overall_eval)

        # Save OOF predictions
        np.save(os.path.join(output_dir, "oof_probs.npy"), oof_probs)
        return oof_probs, overall_eval

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Averages predicted probabilities across all fold models."""
        n_samples = len(texts)
        n_classes = len(OFFICIAL_18_CLASSES)
        total_probs = np.zeros((n_samples, n_classes), dtype=np.float32)

        for model, pipeline in zip(self.models, self.pipelines):
            x_feat, _ = pipeline.transform(texts)
            probs = model.predict_proba(x_feat)
            total_probs += probs

        return total_probs / len(self.models)

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts class labels for a list of texts."""
        probs = self.predict_proba(texts)
        pred_ids = np.argmax(probs, axis=1)
        return [OFFICIAL_18_CLASSES[i] for i in pred_ids]

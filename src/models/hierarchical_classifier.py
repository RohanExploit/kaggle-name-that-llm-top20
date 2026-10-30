"""
Hierarchical Provider-to-Sibling Classifier for Name That LLM.
Stage 1: Predicts the parent AI provider (OpenAI, Google, Anthropic, Qwen, ZAI, Moonshot).
Stage 2: Disambiguates within the predicted provider family to select the exact sibling model.
This directly attacks the primary error mode (58% of errors are right provider, wrong model).
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from src.data import OFFICIAL_18_CLASSES, PROVIDER_MAP, LABEL_TO_ID
from src.metrics import evaluate_predictions, compute_balanced_accuracy


PROVIDERS = sorted(list(set(PROVIDER_MAP.values())))
PROVIDER_TO_ID = {p: i for i, p in enumerate(PROVIDERS)}
ID_TO_PROVIDER = {i: p for i, p in enumerate(PROVIDERS)}

# Siblings per provider
PROVIDER_SIBLINGS = {}
for model, prov in PROVIDER_MAP.items():
    PROVIDER_SIBLINGS.setdefault(prov, []).append(model)


class HierarchicalModel:
    """
    Two-stage hierarchical classifier with calibrated probability output.
    P(model_j) = P(provider_k) * P(model_j | provider_k)
    """

    def __init__(self, c_param: float = 1.5):
        self.c_param = c_param
        self.stage1_model = None
        self.stage2_models = {}

    def fit(self, x_train, y_train_model_ids: np.ndarray):
        # 1. Train Stage 1: Provider classifier
        train_models = [OFFICIAL_18_CLASSES[i] for i in y_train_model_ids]
        train_providers = np.array([PROVIDER_MAP[m] for m in train_models])
        y_provider_ids = np.array([PROVIDER_TO_ID[p] for p in train_providers])

        base_lr = LogisticRegression(
            C=self.c_param,
            max_iter=300,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
        self.stage1_model = CalibratedClassifierCV(base_lr, cv=3)
        self.stage1_model.fit(x_train, y_provider_ids)

        # 2. Train Stage 2: Sibling classifiers for multi-sibling providers
        for prov, siblings in PROVIDER_SIBLINGS.items():
            if len(siblings) == 1:
                continue

            sibling_mask = np.isin(train_models, siblings)
            if sibling_mask.sum() < len(siblings) * 2:
                continue

            x_prov = x_train[sibling_mask]
            sub_y = [siblings.index(m) for m in np.array(train_models)[sibling_mask]]

            sub_lr = LogisticRegression(
                C=self.c_param,
                max_iter=300,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            )
            sub_cal = CalibratedClassifierCV(sub_lr, cv=3)
            sub_cal.fit(x_prov, sub_y)
            self.stage2_models[prov] = (sub_cal, siblings)

        return self

    def predict_proba(self, x_test) -> np.ndarray:
        """Computes joint probabilities P(model) across all 18 classes."""
        n_samples = x_test.shape[0]
        n_classes = len(OFFICIAL_18_CLASSES)
        probs = np.zeros((n_samples, n_classes), dtype=np.float32)

        # Stage 1 probabilities
        p_providers = self.stage1_model.predict_proba(x_test)  # (N, len(PROVIDERS))

        for prov_idx, prov in enumerate(PROVIDERS):
            p_prov = p_providers[:, prov_idx]

            if prov in self.stage2_models:
                sub_cal, siblings = self.stage2_models[prov]
                p_siblings = sub_cal.predict_proba(x_test)  # (N, len(siblings))
                for sib_idx, model_name in enumerate(siblings):
                    global_class_idx = LABEL_TO_ID[model_name]
                    probs[:, global_class_idx] = p_prov * p_siblings[:, sib_idx]
            else:
                # Single-model provider (e.g. Qwen, Moonshot)
                single_model = PROVIDER_SIBLINGS[prov][0]
                global_class_idx = LABEL_TO_ID[single_model]
                probs[:, global_class_idx] = p_prov

        # Normalize to ensure valid probability distribution
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, x_test) -> np.ndarray:
        probs = self.predict_proba(x_test)
        return np.argmax(probs, axis=1)

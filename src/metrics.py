"""
Evaluation metrics and diagnostic reporting for Name That LLM.
Competition Metric: Balanced Accuracy over K=18 candidate models.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the competition metric: balanced accuracy (macro-averaged recall)."""
    return float(balanced_accuracy_score(y_true, y_pred))


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    """Computes per-class recall for all 18 classes."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    recalls = {}
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        recalls[name] = float(cm[i, i] / total) if total > 0 else 0.0
    return recalls


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    provider_map: Dict[str, str] = None,
) -> Dict:
    """
    Comprehensive evaluation returning:
    - balanced_accuracy (competition official metric)
    - standard accuracy
    - per-class recall
    - provider-level accuracy (if provider_map is supplied)
    - sibling error fraction (percentage of errors where provider was correct but sibling model was misclassified)
    """
    bal_acc = compute_balanced_accuracy(y_true, y_pred)
    acc = float((y_true == y_pred).mean())
    recalls = per_class_recall(y_true, y_pred, class_names)

    result = {
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "per_class_recall": recalls,
    }

    if provider_map:
        # Map class index -> provider name
        class_to_provider = [provider_map.get(c, "unknown") for c in class_names]
        true_providers = np.array([class_to_provider[i] for i in y_true])
        pred_providers = np.array([class_to_provider[i] for i in y_pred])

        provider_acc = float((true_providers == pred_providers).mean())
        mistakes = y_true != y_pred
        num_mistakes = int(mistakes.sum())

        if num_mistakes > 0:
            sibling_mistakes = int((mistakes & (true_providers == pred_providers)).sum())
            sibling_ratio = sibling_mistakes / num_mistakes
        else:
            sibling_mistakes = 0
            sibling_ratio = 0.0

        result["provider_accuracy"] = provider_acc
        result["total_mistakes"] = num_mistakes
        result["sibling_mistakes"] = sibling_mistakes
        result["sibling_error_fraction"] = sibling_ratio

    return result


def print_evaluation_summary(eval_results: Dict):
    """Prints a clean, formatted terminal summary of the evaluation."""
    print("=" * 65)
    print(f"OFFICIAL COMPETITION METRIC - BALANCED ACCURACY: {eval_results['balanced_accuracy']:.4f}")
    print(f"Raw Accuracy:                                    {eval_results['accuracy']:.4f}")
    if "provider_accuracy" in eval_results:
        print(f"Provider/Lab Accuracy:                           {eval_results['provider_accuracy']:.4f}")
        print(
            f"Sibling Error Ratio (Right Lab, Wrong Model):   "
            f"{eval_results['sibling_error_fraction']:.1%} "
            f"({eval_results['sibling_mistakes']}/{eval_results['total_mistakes']})"
        )
    print("-" * 65)
    print(f"{'Model Name':<28} | {'Recall':<10}")
    print("-" * 65)
    for model_name, rec in eval_results["per_class_recall"].items():
        print(f"{model_name:<28} | {rec:.4f}")
    print("=" * 65)

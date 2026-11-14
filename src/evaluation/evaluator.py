"""Evaluation metrics: Macro F1, Weighted F1, and Per-class Precision."""
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np

def compute_nlp_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": [float(x) for x in f1]
    }

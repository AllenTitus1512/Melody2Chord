"""Evaluation utilities that report honest baseline model performance."""
from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def evaluate_model(model, X_test, y_test) -> dict[str, Any]:
    """Calculate and print standard multi-class classification metrics."""
    predictions = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    labels = list(model.classes_)
    results = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels), "labels": labels,
        "classification_report": classification_report(y_test, predictions, labels=labels, zero_division=0),
    }
    print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Precision (weighted): {results['precision'] * 100:.2f}%")
    print(f"Recall (weighted): {results['recall'] * 100:.2f}%")
    print(f"F1-score (weighted): {results['f1'] * 100:.2f}%")
    print("\nClassification Report:")
    print(results["classification_report"])
    print("Confusion Matrix (rows=true, columns=predicted):")
    print(results["confusion_matrix"])
    return results

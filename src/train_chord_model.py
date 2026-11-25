"""Compare the original Random Forest with a musical-feature v2 on one song split."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = ROOT / "data" / "model_results" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit

try:
    from src.feature_extraction import (FeatureExtractionError, TRAINING_FEATURE_NAMES,
                                        V2_TRAINING_FEATURE_NAMES, extract_dataset_features,
                                        extract_dataset_features_v2)
except ModuleNotFoundError:
    from feature_extraction import (FeatureExtractionError, TRAINING_FEATURE_NAMES,
                                    V2_TRAINING_FEATURE_NAMES, extract_dataset_features,
                                    extract_dataset_features_v2)

DATASET_PATH = ROOT / "data" / "training" / "melody_chord_dataset.csv"
RESULTS_DIR = ROOT / "data" / "model_results"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate each metric directly from sklearn using the same predictions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _format_metrics(metrics: dict[str, float]) -> str:
    return "\n".join(f"{name}: {metrics[name]:.6f}" for name in
                     ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATASET_PATH)
    required = ("song_id", "melody_notes", "note_frequencies", "note_durations", "musical_key", "chord_label")
    missing = [column for column in required if column not in frame.columns]
    if missing or frame[list(required)].isna().any().any():
        raise ValueError(f"Dataset has missing required columns or values: {missing}")

    old_vectors, new_vectors, kept_indices = [], [], []
    for index, row in frame.iterrows():
        try:
            old_vectors.append(extract_dataset_features(row))
            new_vectors.append(extract_dataset_features_v2(row))
            kept_indices.append(index)
        except FeatureExtractionError as error:
            print(f"Skipping CSV data row {index + 2}: {error}")
    usable = frame.loc[kept_indices].reset_index(drop=True)
    if not len(usable):
        raise ValueError("No valid rows remain after feature extraction.")
    X_old = pd.DataFrame(np.vstack(old_vectors), columns=TRAINING_FEATURE_NAMES)
    X_new = pd.DataFrame(np.vstack(new_vectors), columns=V2_TRAINING_FEATURE_NAMES)
    if not np.isfinite(X_old.to_numpy()).all() or not np.isfinite(X_new.to_numpy()).all():
        raise ValueError("Feature matrix contains non-finite values.")
    y = usable["chord_label"].astype(str).reset_index(drop=True)
    groups = usable["song_id"].astype(str).reset_index(drop=True)
    forbidden = {"song_id", "chord_label", "start_time", "end_time"}
    if forbidden & set(V2_TRAINING_FEATURE_NAMES):
        raise RuntimeError("Leakage check failed: a forbidden field is an input feature.")

    split = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(split.split(X_new, y, groups=groups))
    train_songs, test_songs = set(groups.iloc[train_idx]), set(groups.iloc[test_idx])
    overlap = train_songs & test_songs
    if overlap:
        raise RuntimeError(f"Song leakage detected: {sorted(overlap)}")
    X_old_train, X_old_test = X_old.iloc[train_idx], X_old.iloc[test_idx]
    X_train, X_test = X_new.iloc[train_idx], X_new.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    labels = sorted(y.unique())

    old_model_path = MODELS_DIR / "chord_random_forest.joblib"
    if not old_model_path.is_file():
        raise FileNotFoundError(f"Original baseline model not found: {old_model_path}")
    old_model = joblib.load(old_model_path)
    if old_model.n_features_in_ != len(TRAINING_FEATURE_NAMES):
        raise ValueError("Original model feature count does not match the preserved baseline extractor.")
    # Prediction is deterministic and needs no worker pool; avoid platform
    # thread restrictions while preserving the original model file unchanged.
    old_model.n_jobs = 1
    old_predictions = old_model.predict(X_old_test)
    old_metrics = _metrics(y_test, old_predictions)
    old_report = classification_report(y_test, old_predictions, labels=labels, zero_division=0)

    print(f"Training songs: {len(train_songs)}")
    print(f"Testing songs: {len(test_songs)}")
    print(f"Training rows: {len(train_idx)}")
    print(f"Testing rows: {len(test_idx)}")
    print("Chord counts (full dataset):", y.value_counts().sort_index().to_dict())
    print("Chord counts (training set):", y_train.value_counts().sort_index().to_dict())
    print("Chord counts (test set):", y_test.value_counts().sort_index().to_dict())

    new_model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                       class_weight="balanced", n_jobs=-1)
    print("Training started...")
    new_model.fit(X_train, y_train)
    print("Training completed...")
    new_predictions = new_model.predict(X_test)
    new_metrics = _metrics(y_test, new_predictions)
    new_report = classification_report(y_test, new_predictions, labels=labels, zero_division=0)
    common_classes = [label for label in labels if label not in {"C7", "G7"}]
    common_report = classification_report(y_test, new_predictions, labels=common_classes, zero_division=0)

    importance = pd.DataFrame({"feature": V2_TRAINING_FEATURE_NAMES,
                               "importance": new_model.feature_importances_}).sort_values(
                                   "importance", ascending=False, kind="stable")
    top15 = importance.head(15).reset_index(drop=True)
    top15.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    fig, axis = plt.subplots(figsize=(10, 7))
    ordered = top15.sort_values("importance", ascending=True)
    axis.barh(ordered["feature"], ordered["importance"], color="steelblue")
    axis.set_xlabel("Feature importance")
    axis.set_title("Melody2Chord Random Forest Feature Importance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)

    # Preserve the verified v1 outputs in a comparison report before replacing current reports.
    comparison = (
        "MELODY2CHORD BASELINE COMPARISON\n\n"
        "OLD BASELINE — 12-class evaluation\n" + _format_metrics(old_metrics) +
        "\n\n" + old_report +
        "\n\nNEW MODEL — 12-class evaluation\n" + _format_metrics(new_metrics) +
        "\n\n" + new_report +
        "\n\nNEW MODEL — 10-class diagnostic report (C7 and G7 omitted from displayed labels)\n" + common_report
    )
    (RESULTS_DIR / "baseline_comparison.txt").write_text(comparison, encoding="utf-8")
    (RESULTS_DIR / "classification_report.txt").write_text(
        "NEW MODEL — full 12-class classification report\n\n" + new_report, encoding="utf-8")
    confusion_labels = labels
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_test, new_predictions, labels=confusion_labels)
    pd.DataFrame(confusion, index=confusion_labels, columns=confusion_labels).rename_axis("actual").to_csv(
        RESULTS_DIR / "confusion_matrix.csv")
    fig, axis = plt.subplots(figsize=(10, 8))
    display = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    fig.colorbar(display, ax=axis)
    axis.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels,
             yticklabels=labels, xlabel="Predicted chord", ylabel="Actual chord",
             title="Melody2Chord Random Forest Confusion Matrix")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    v2_path = MODELS_DIR / "chord_random_forest_v2.joblib"
    joblib.dump(new_model, v2_path)
    key_encoding = {
        "key_root_pitch_class": "C=0, C#=1, ..., B=11",
        "key_mode": {"major": 0, "minor": 1},
        "minor_scale_representation": "natural minor (Aeolian): semitones 0,2,3,5,7,8,10",
    }
    model_metadata = {"model_type": "RandomForestClassifier", "feature_names": list(V2_TRAINING_FEATURE_NAMES),
                      "chord_classes": labels, "key_encoding": key_encoding}
    (MODELS_DIR / "chord_model_v2_metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    eval_metadata = {
        "model_type": "RandomForestClassifier", "dataset_path": "data/training/melody_chord_dataset.csv",
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        "training_song_count": len(train_songs), "testing_song_count": len(test_songs),
        "training_row_count": len(train_idx), "testing_row_count": len(test_idx),
        "feature_count": len(V2_TRAINING_FEATURE_NAMES), "feature_names": list(V2_TRAINING_FEATURE_NAMES),
        "chord_classes": labels, **new_metrics, "train_test_song_overlap": sorted(overlap),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS_DIR / "model_metadata.json").write_text(json.dumps(eval_metadata, indent=2), encoding="utf-8")

    print("\n========================================")
    print("MELODY2CHORD BASELINE VS V2")
    print("========================================")
    print("OLD BASELINE:")
    print(_format_metrics(old_metrics))
    print("\nNEW MODEL:")
    print(_format_metrics(new_metrics))
    print(f"\nTraining songs: {len(train_songs)} | Testing songs: {len(test_songs)}")
    print(f"Training rows: {len(train_idx)} | Testing rows: {len(test_idx)}")
    print(f"V2 features: {len(V2_TRAINING_FEATURE_NAMES)}")
    print("Train/test song overlap: 0")
    print("\nTop features:")
    for rank, item in enumerate(top15.itertuples(index=False), start=1):
        print(f"{rank}. {item.feature}: {item.importance:.6f}")
    print("\n12-class classification report:\n" + new_report)
    print("10-class report excluding C7/G7 (diagnostic only):\n" + common_report)
    print(f"V2 model saved: {v2_path.relative_to(ROOT)}")
    print("========================================")


if __name__ == "__main__":
    main()

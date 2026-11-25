"""Train and evaluate an offline, song-held-out LSTM on POP909 segments."""
from __future__ import annotations

import json
import os
import re
import random
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "model_results"
MODELS_DIR = ROOT / "models"
MPL_CONFIG_DIR = RESULTS_DIR / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
TEMP_DIR = RESULTS_DIR / ".tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)
tempfile.tempdir = str(TEMP_DIR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

try:
    from src.lstm_feature_extraction import (CONTINUOUS_FEATURE_INDICES, FEATURE_NAMES,
                                             LSTMFeatureError, extract_melody_sequence,
                                             fit_sequence_scaler, scale_and_pad_sequence)
except ModuleNotFoundError:
    from lstm_feature_extraction import (CONTINUOUS_FEATURE_INDICES, FEATURE_NAMES,
                                         LSTMFeatureError, extract_melody_sequence,
                                         fit_sequence_scaler, scale_and_pad_sequence)
try:
    from src.lstm_layers import MaskAwareLSTM
except ModuleNotFoundError:
    from lstm_layers import MaskAwareLSTM

DATASET_PATH = ROOT / "data" / "training" / "melody_chord_dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10
CHORD_CLASSES = ("A", "Am", "Bm", "C", "C7", "D", "Dm", "E", "Em", "F", "G", "G7")
LABEL_MAPPING = {label: index for index, label in enumerate(CHORD_CLASSES)}
REQUIRED_COLUMNS = ("song_id", "melody_notes", "note_frequencies", "note_durations", "musical_key", "chord_label")


def _make_split(groups: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row indices for train/validation/test with deterministic group splits."""
    row_indices = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    development_idx, test_idx = next(outer.split(row_indices, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)
    train_rel, validation_rel = next(inner.split(
        development_idx, groups=groups.iloc[development_idx].reset_index(drop=True)))
    train_idx = development_idx[train_rel]
    validation_idx = development_idx[validation_rel]
    return train_idx, validation_idx, test_idx


def _transform_sequences(
    sequences: list[np.ndarray], indices: np.ndarray, sequence_length: int, scaler
) -> np.ndarray:
    """Apply the shared scaler and sequence post-padding for a dataset partition."""
    return np.stack([scale_and_pad_sequence(sequences[int(index)], scaler, sequence_length)
                     for index in indices])


def _read_rf_metrics() -> tuple[dict[str, float], dict[str, float]]:
    """Read the preserved v1 comparison and v2 evaluation metadata without rewriting either."""
    comparison_path = RESULTS_DIR / "baseline_comparison.txt"
    comparison = comparison_path.read_text(encoding="utf-8")
    old_section = re.search(r"OLD BASELINE.*?(?=NEW MODEL)", comparison, flags=re.DOTALL)
    if old_section is None:
        raise ValueError("Could not find the saved Random Forest V1 metrics in baseline_comparison.txt.")
    old_metrics: dict[str, float] = {}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        match = re.search(rf"\b{key}:\s*([0-9.]+)", old_section.group(0))
        if match is None:
            raise ValueError(f"Saved V1 comparison is missing {key}.")
        old_metrics[key] = float(match.group(1))
    v2 = json.loads((RESULTS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    v2_metrics = {key: float(v2[key]) for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}
    return old_metrics, v2_metrics


def _plot_history(history: dict[str, list[float]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["loss"], label="Training Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set(title="LSTM Training and Validation Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(history["accuracy"], label="Training Accuracy")
    axes[1].plot(history["val_accuracy"], label="Validation Accuracy")
    axes[1].set(title="LSTM Training and Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "lstm_training_history.png", dpi=160)
    plt.close(figure)


def main() -> None:
    if int(tf.__version__.split(".")[0]) < 2:
        raise RuntimeError(f"TensorFlow 2.x is required; found {tf.__version__}.")
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(DATASET_PATH)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Dataset contains missing required values; stop before training.")
    observed_classes = set(frame["chord_label"].astype(str))
    if observed_classes != set(CHORD_CLASSES):
        raise ValueError(f"Expected the established 12 chord classes; found {sorted(observed_classes)}")

    sequences: list[np.ndarray] = []
    bad_rows: list[str] = []
    for index, row in frame.iterrows():
        try:
            sequences.append(extract_melody_sequence(row))
        except LSTMFeatureError as error:
            bad_rows.append(f"CSV row {index + 2}: {error}")
            sequences.append(np.empty((0, len(FEATURE_NAMES)), dtype=np.float32))
    if bad_rows:
        raise ValueError("Sequence extraction failed; no rows were removed:\n" + "\n".join(bad_rows[:20]))

    song_ids = frame["song_id"].astype(str).reset_index(drop=True)
    train_idx, validation_idx, test_idx = _make_split(song_ids)
    train_songs = set(song_ids.iloc[train_idx])
    validation_songs = set(song_ids.iloc[validation_idx])
    test_songs = set(song_ids.iloc[test_idx])
    train_validation_overlap = train_songs & validation_songs
    train_test_overlap = train_songs & test_songs
    validation_test_overlap = validation_songs & test_songs
    if train_validation_overlap or train_test_overlap or validation_test_overlap:
        raise RuntimeError("Song overlap detected; stopping before training.")
    if len(test_songs) != 161:
        raise RuntimeError(f"Expected the same 161-song Random Forest test split; got {len(test_songs)} songs.")

    labels = frame["chord_label"].astype(str).map(LABEL_MAPPING).to_numpy(dtype=np.int32)
    y_train, y_validation, y_test = labels[train_idx], labels[validation_idx], labels[test_idx]
    train_lengths = np.asarray([len(sequences[index]) for index in train_idx], dtype=int)
    raw_maximum_length = int(train_lengths.max())
    percentile_95 = float(np.percentile(train_lengths, 95))
    sequence_length = min(128, max(1, int(np.ceil(percentile_95))))
    truncated = np.asarray([len(sequence) > sequence_length for sequence in sequences], dtype=bool)
    truncated_count = int(truncated.sum())
    truncated_percentage = float(100.0 * truncated_count / len(sequences))

    # Fit the scaler only on usable, capped training notes. Categorical one-hot
    # and binary features are left as-is; only continuous feature columns scale.
    scaler = fit_sequence_scaler([sequences[index] for index in train_idx], sequence_length)
    X_train = _transform_sequences(sequences, train_idx, sequence_length, scaler)
    X_validation = _transform_sequences(sequences, validation_idx, sequence_length, scaler)
    X_test = _transform_sequences(sequences, test_idx, sequence_length, scaler)

    present_training_classes = np.unique(y_train)
    class_values = compute_class_weight(class_weight="balanced", classes=present_training_classes, y=y_train)
    class_weights = {int(label): float(weight) for label, weight in zip(present_training_classes, class_values)}
    if len(present_training_classes) != len(CHORD_CLASSES):
        raise ValueError("At least one expected chord class is absent from training; cannot establish all-class weights.")

    input_layer = tf.keras.Input(shape=(sequence_length, len(FEATURE_NAMES)), name="melody_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0, name="padding_mask")(input_layer)
    lstm_layer = MaskAwareLSTM(64, name="lstm_64")
    x = lstm_layer(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout_30")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_32")(x)
    x = tf.keras.layers.Dropout(0.20, name="dropout_20")(x)
    output_layer = tf.keras.layers.Dense(len(CHORD_CLASSES), activation="softmax", name="chord_probabilities")(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="melody2chord_lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Maximum raw sequence length (training songs): {raw_maximum_length}")
    print(f"Chosen sequence length: {sequence_length}")
    print(f"Number of sequences truncated: {truncated_count} ({truncated_percentage:.2f}%)")
    print(f"Training songs: {len(train_songs)}")
    print(f"Validation songs: {len(validation_songs)}")
    print(f"Testing songs: {len(test_songs)}")
    print(f"Training rows: {len(train_idx)}")
    print(f"Validation rows: {len(validation_idx)}")
    print(f"Testing rows: {len(test_idx)}")
    print(f"Train/validation overlap: {len(train_validation_overlap)}")
    print(f"Train/test overlap: {len(train_test_overlap)}")
    print(f"Validation/test overlap: {len(validation_test_overlap)}")
    print("Training class weights:", {CHORD_CLASSES[key]: value for key, value in class_weights.items()})
    print("Training started...")
    fitted_history = model.fit(
        X_train, y_train, validation_data=(X_validation, y_validation),
        epochs=50, batch_size=64, class_weight=class_weights, callbacks=callbacks, verbose=2,
    )
    print("Training completed...")

    probabilities = model.predict(X_test, batch_size=64, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    accuracy = float(accuracy_score(y_test, predicted_labels))
    balanced_accuracy = float(balanced_accuracy_score(y_test, predicted_labels))
    macro_f1 = float(f1_score(y_test, predicted_labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, predicted_labels, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_test, predicted_labels, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test, predicted_labels, average="macro", zero_division=0))
    report = classification_report(y_test, predicted_labels, labels=np.arange(len(CHORD_CLASSES)),
                                   target_names=CHORD_CLASSES, zero_division=0)
    matrix = confusion_matrix(y_test, predicted_labels, labels=np.arange(len(CHORD_CLASSES)))
    low_support = {CHORD_CLASSES[index]: int(np.sum(y_test == index)) for index in range(len(CHORD_CLASSES))
                   if np.sum(y_test == index) < 10}
    low_support_validation = {
        CHORD_CLASSES[index]: int(np.sum(y_validation == index)) for index in range(len(CHORD_CLASSES))
        if np.sum(y_validation == index) < 10
    }

    (RESULTS_DIR / "lstm_classification_report.txt").write_text(
        "LSTM test-set classification report (unseen songs only)\n\n" + report, encoding="utf-8")
    pd.DataFrame(matrix, index=CHORD_CLASSES, columns=CHORD_CLASSES).rename_axis("actual").to_csv(
        RESULTS_DIR / "lstm_confusion_matrix.csv")
    figure, axis = plt.subplots(figsize=(10, 8))
    displayed = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(displayed, ax=axis)
    axis.set(xticks=np.arange(len(CHORD_CLASSES)), yticks=np.arange(len(CHORD_CLASSES)),
             xticklabels=CHORD_CLASSES, yticklabels=CHORD_CLASSES,
             xlabel="Predicted Chord", ylabel="Actual Chord",
             title="Melody2Chord LSTM Confusion Matrix")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "lstm_confusion_matrix.png", dpi=160)
    plt.close(figure)
    history_values = {key: [float(value) for value in values]
                      for key, values in fitted_history.history.items()}
    _plot_history(history_values)

    model_path = MODELS_DIR / "chord_lstm.keras"
    model.save(model_path)
    scaler_path = MODELS_DIR / "chord_lstm_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    class_weight_by_name = {CHORD_CLASSES[index]: float(weight) for index, weight in class_weights.items()}
    metadata = {
        "model_type": "TensorFlow Keras LSTM", "tensorflow_version": tf.__version__,
        "dataset_path": "data/training/melody_chord_dataset.csv", "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE, "validation_size_of_development": VALIDATION_SIZE,
        "training_song_count": len(train_songs), "validation_song_count": len(validation_songs),
        "testing_song_count": len(test_songs), "training_row_count": len(train_idx),
        "validation_row_count": len(validation_idx), "testing_row_count": len(test_idx),
        "feature_names": list(FEATURE_NAMES), "feature_count": len(FEATURE_NAMES),
        "continuous_feature_names": [FEATURE_NAMES[index] for index in CONTINUOUS_FEATURE_INDICES],
        "max_sequence_length": sequence_length, "maximum_raw_training_sequence_length": raw_maximum_length,
        "training_length_95th_percentile": percentile_95, "truncated_sequence_count": truncated_count,
        "truncated_sequence_percentage": truncated_percentage,
        "chord_classes": list(CHORD_CLASSES), "chord_label_mapping": LABEL_MAPPING,
        "class_weights": class_weight_by_name, "scaler_path": "models/chord_lstm_scaler.joblib",
        "accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "macro_precision": macro_precision, "macro_recall": macro_recall,
        "train_validation_overlap": sorted(train_validation_overlap),
        "train_test_overlap": sorted(train_test_overlap),
        "validation_test_overlap": sorted(validation_test_overlap),
        "low_support_test_classes": low_support,
        "low_support_validation_classes": low_support_validation,
        "architecture": ["Masking(0.0)", "LSTM(64)", "Dropout(0.30)",
                         "Dense(32, relu)", "Dropout(0.20)", "Dense(12, softmax)"],
        "training_epochs_completed": len(history_values.get("loss", [])),
    }
    (MODELS_DIR / "chord_lstm_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (RESULTS_DIR / "lstm_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    old_metrics, v2_metrics = _read_rf_metrics()
    comparison = {
        "Random Forest V1": old_metrics,
        "Random Forest V2": v2_metrics,
        "LSTM": {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
                 "macro_f1": macro_f1, "weighted_f1": weighted_f1},
    }
    comparison_lines = ["MODEL COMPARISON", "", f"{'Model':<20} {'Accuracy':>10} {'Balanced Acc':>14} {'Macro F1':>10} {'Weighted F1':>13}"]
    for name, metrics in comparison.items():
        comparison_lines.append(
            f"{name:<20} {metrics['accuracy']:>10.4f} {metrics['balanced_accuracy']:>14.4f} "
            f"{metrics['macro_f1']:>10.4f} {metrics['weighted_f1']:>13.4f}")
    (RESULTS_DIR / "model_comparison.txt").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")

    print("\n========================================")
    print("MELODY2CHORD LSTM RESULTS")
    print("========================================")
    print(f"Training songs: {len(train_songs)}")
    print(f"Validation songs: {len(validation_songs)}")
    print(f"Testing songs: {len(test_songs)}")
    print(f"Training rows: {len(train_idx)}")
    print(f"Validation rows: {len(validation_idx)}")
    print(f"Testing rows: {len(test_idx)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Features per timestep: {len(FEATURE_NAMES)}")
    print(f"Truncated sequences: {truncated_count} ({truncated_percentage:.2f}%)")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Train/validation overlap: {len(train_validation_overlap)}")
    print(f"Train/test overlap: {len(train_test_overlap)}")
    print(f"Validation/test overlap: {len(validation_test_overlap)}")
    print("Model: LSTM")
    print("Saved model: models/chord_lstm.keras")
    print("\nPer-class test report:\n" + report)
    print("Low-support test classes (<10 rows):", low_support)
    print("Low-support validation classes (<10 rows):", low_support_validation)
    print("\n" + "\n".join(comparison_lines))
    print("========================================")


if __name__ == "__main__":
    main()

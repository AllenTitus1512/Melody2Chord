"""Load and run the saved Melody2Chord LSTM on live melody sequences."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "chord_lstm.keras"
METADATA_PATH = ROOT / "models" / "chord_lstm_metadata.json"
SCALER_PATH = ROOT / "models" / "chord_lstm_scaler.joblib"
WINDOW_SECONDS = 1.8  # Median annotated chord interval in POP909 training data.
MIN_WINDOW_SECONDS = 0.952381  # 10th percentile annotated chord interval.
MIN_NOTES_PER_WINDOW = 2  # 25th percentile training segment has two melody notes.
OVERLAP_EPSILON = 1e-6
LOGGER = logging.getLogger(__name__)

try:
    from src.lstm_feature_extraction import FEATURE_NAMES, LSTMFeatureError, extract_melody_sequence, scale_and_pad_sequence
except ModuleNotFoundError:
    from lstm_feature_extraction import FEATURE_NAMES, LSTMFeatureError, extract_melody_sequence, scale_and_pad_sequence


class ChordPredictionError(Exception):
    """A safe-to-display error while preparing or predicting chord suggestions."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def _configure_tf_environment() -> None:
    results_temp = ROOT / "data" / "model_results" / ".tmp"
    results_temp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMP", str(results_temp))
    os.environ.setdefault("TEMP", str(results_temp))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    tempfile.tempdir = os.environ.get("TEMP")


@lru_cache(maxsize=1)
def _load_assets() -> tuple[Any, dict[str, Any], Any | None]:
    """Load model assets once per process; this never trains or writes a model."""
    if not MODEL_PATH.is_file():
        raise ChordPredictionError("The trained chord model is unavailable on this server.", 503)
    if not METADATA_PATH.is_file():
        raise ChordPredictionError("The trained chord model metadata is unavailable on this server.", 503)
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if metadata.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Feature layout differs from the saved model metadata.")
        classes = metadata.get("chord_classes")
        mapping = metadata.get("chord_label_mapping")
        length = int(metadata.get("max_sequence_length", 0))
        if not isinstance(classes, list) or not classes or not isinstance(mapping, dict):
            raise ValueError("Saved chord class mapping is incomplete.")
        if len(classes) != len(mapping) or any(mapping.get(label) != i for i, label in enumerate(classes)):
            raise ValueError("Saved chord class mapping is inconsistent.")
        if int(metadata.get("feature_count", -1)) != len(FEATURE_NAMES) or length < 1:
            raise ValueError("Saved feature count or sequence length is invalid.")

        _configure_tf_environment()
        import tensorflow as tf
        # Register the custom layer used when this model was saved.
        try:
            from src.lstm_layers import MaskAwareLSTM  # noqa: F401
        except ModuleNotFoundError:
            from lstm_layers import MaskAwareLSTM  # noqa: F401
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        expected_shape = tuple(model.input_shape[1:])
        if expected_shape != (length, len(FEATURE_NAMES)) or int(model.output_shape[-1]) != len(classes):
            raise ValueError("Saved model input/output dimensions do not match metadata.")

        scaler = None
        scaler_required = bool(metadata.get("scaler_path"))
        if scaler_required:
            if not SCALER_PATH.is_file():
                raise ChordPredictionError("The scaler required by the trained chord model is unavailable.", 503)
            import joblib
            scaler = joblib.load(SCALER_PATH)
            if int(getattr(scaler, "n_features_in_", -1)) != 6:
                raise ValueError("Saved scaler does not match the six continuous melody features.")
        return model, metadata, scaler
    except ChordPredictionError:
        raise
    except Exception as error:
        LOGGER.exception("Could not load LSTM prediction assets")
        raise ChordPredictionError("The trained chord model could not be loaded.", 503) from error


def clear_model_cache() -> None:
    """Clear cached assets, primarily for controlled test isolation."""
    _load_assets.cache_clear()


def is_supported_musical_key(musical_key: Any) -> bool:
    """Accept only the pitch-name and major/minor format emitted by key detection."""
    if not isinstance(musical_key, str):
        return False
    parts = musical_key.strip().split()
    if len(parts) != 2:
        return False
    try:
        from src.lstm_feature_extraction import PITCH_CLASS_INDEX
    except ModuleNotFoundError:
        from lstm_feature_extraction import PITCH_CLASS_INDEX
    return parts[0] in PITCH_CLASS_INDEX and parts[1].lower() in {"major", "minor"}


def make_prediction_windows(timeline: Sequence[dict[str, Any]], window_seconds: float = WINDOW_SECONDS) -> list[dict[str, Any]]:
    """Group note events into fixed 1.8-second, non-overlapping melody windows."""
    if not timeline:
        raise ChordPredictionError("No melody notes were detected. Extract a clear vocal melody first.")
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ChordPredictionError("Prediction window configuration is invalid.")
    valid_events = []
    for item in timeline:
        try:
            start = float(item["start"])
            end = float(item["end"])
            frequency = float(item["frequency"])
            note = str(item["note"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite([start, end, frequency]).all() and end > start and frequency > 0 and note):
            continue
        valid_events.append({"start": start, "end": end, "frequency": frequency, "note": note})
    if not valid_events:
        raise ChordPredictionError("No usable melody notes were detected in this recording.")

    first = min(event["start"] for event in valid_events)
    last = max(event["end"] for event in valid_events)
    result = []
    index = 0
    while first + index * window_seconds < last:
        start = first + index * window_seconds
        end = min(start + window_seconds, last)
        span = end - start
        if span + OVERLAP_EPSILON >= MIN_WINDOW_SECONDS:
            notes, frequencies, durations = [], [], []
            for event in valid_events:
                overlap_start = max(event["start"], start)
                overlap_end = min(event["end"], end)
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > OVERLAP_EPSILON:
                    notes.append(event["note"])
                    frequencies.append(event["frequency"])
                    durations.append(overlap_duration)
            if len(notes) >= MIN_NOTES_PER_WINDOW:
                result.append({"start": start, "end": end, "notes": notes,
                               "frequencies": frequencies, "durations": durations})
        index += 1
    if not result:
        raise ChordPredictionError("This melody does not contain enough notes in a prediction window. Try a longer vocal phrase.")
    return result


def predict_chords(
    melody_notes: Sequence[str],
    note_frequencies: Sequence[float],
    note_durations: Sequence[float],
    musical_key: str,
) -> dict[str, Any]:
    """Predict a chord distribution from one already-segmented melody window."""
    if isinstance(musical_key, str) is False or not musical_key.strip():
        raise ChordPredictionError("A detected musical key is required before chord suggestions can be generated.")
    if not isinstance(melody_notes, Sequence) or isinstance(melody_notes, (str, bytes)) or len(melody_notes) < MIN_NOTES_PER_WINDOW:
        raise ChordPredictionError("At least two melody notes are required in a prediction window.")
    if not (len(melody_notes) == len(note_frequencies) == len(note_durations)):
        raise ChordPredictionError("Melody note, frequency, and duration counts do not match.")
    row = {"melody_notes": ",".join(str(note) for note in melody_notes),
           "note_frequencies": ",".join(str(value) for value in note_frequencies),
           "note_durations": ",".join(str(value) for value in note_durations),
           "musical_key": musical_key}
    try:
        sequence = extract_melody_sequence(row)
    except LSTMFeatureError as error:
        raise ChordPredictionError(f"The melody or detected key is not supported: {error}") from error
    model, metadata, scaler = _load_assets()
    try:
        if scaler is None:
            # Metadata without a scaler means the training pipeline used raw features.
            padded = np.zeros((int(metadata["max_sequence_length"]), len(FEATURE_NAMES)), dtype=np.float32)
            capped = sequence[:len(padded)]
            padded[:len(capped)] = capped
        else:
            padded = scale_and_pad_sequence(sequence, scaler, int(metadata["max_sequence_length"]))
        output = np.asarray(model.predict(padded[np.newaxis, ...], verbose=0), dtype=float)
        classes = metadata["chord_classes"]
        if output.shape != (1, len(classes)) or not np.isfinite(output).all():
            raise ValueError("Model returned empty, non-finite, or incorrectly shaped probabilities.")
        probabilities = output[0]
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
            raise ValueError("Model probability output is invalid.")
        ranked = np.argsort(probabilities)[::-1]
        predicted_index = int(ranked[0])
        return {"chord": classes[predicted_index], "probability": float(probabilities[predicted_index]),
                "alternatives": [{"chord": classes[int(i)], "probability": float(probabilities[int(i)])}
                                 for i in ranked[1:4]],
                "probabilities": {classes[i]: float(probabilities[i]) for i in range(len(classes))}}
    except Exception as error:
        LOGGER.exception("LSTM inference failed")
        raise ChordPredictionError("Chord suggestions could not be generated for this melody window.", 500) from error


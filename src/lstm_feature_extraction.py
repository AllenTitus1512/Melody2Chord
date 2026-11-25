"""Deterministic note-by-note feature extraction for LSTM chord prediction."""
from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PITCH_CLASS_SAFE_NAMES = ("C", "Csharp", "D", "Dsharp", "E", "F", "Fsharp", "G", "Gsharp", "A", "Asharp", "B")
PITCH_CLASS_INDEX = {name: index for index, name in enumerate(PITCH_CLASS_NAMES)}
PITCH_CLASS_INDEX.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})
NOTE_RE = re.compile(r"^([A-G])([#b]?)(-?\d+)$")
CONTINUOUS_FEATURES = (
    "midi_pitch", "frequency_hz", "normalized_midi_pitch", "note_duration",
    "pitch_interval_from_previous", "absolute_pitch_interval",
)
FEATURE_NAMES = (
    *CONTINUOUS_FEATURES,
    *(f"pitch_class_{name}" for name in PITCH_CLASS_SAFE_NAMES),
    *(f"relative_pitch_class_to_key_{name}" for name in PITCH_CLASS_SAFE_NAMES),
    "repeated_from_previous", "interval_ascending", "interval_descending",
    *(f"key_root_{name}" for name in PITCH_CLASS_SAFE_NAMES),
    "key_mode_minor",
)
CONTINUOUS_FEATURE_INDICES = tuple(range(len(CONTINUOUS_FEATURES)))


class LSTMFeatureError(ValueError):
    """Raised when a dataset row cannot be represented as a valid note sequence."""


def fit_sequence_scaler(training_sequences: list[np.ndarray], max_sequence_length: int) -> StandardScaler:
    """Fit the continuous-feature scaler on capped training notes only."""
    if not training_sequences or max_sequence_length < 1:
        raise LSTMFeatureError("Training sequences and a positive maximum length are required to fit scaling.")
    continuous = np.asarray(CONTINUOUS_FEATURE_INDICES, dtype=int)
    note_values = np.concatenate([sequence[:max_sequence_length, continuous] for sequence in training_sequences], axis=0)
    if not np.isfinite(note_values).all():
        raise LSTMFeatureError("Training note features contain non-finite values.")
    return StandardScaler().fit(note_values)


def scale_and_pad_sequence(sequence: np.ndarray, scaler: StandardScaler, max_sequence_length: int) -> np.ndarray:
    """Use the training scaler and the LSTM's truncation/post-padding convention."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or values.shape[0] == 0:
        raise LSTMFeatureError("Expected a nonempty note-by-feature sequence with the trained feature layout.")
    if max_sequence_length < 1:
        raise LSTMFeatureError("Maximum sequence length must be positive.")
    values = values[:max_sequence_length].copy()
    if not np.isfinite(values).all():
        raise LSTMFeatureError("Sequence contains non-finite values.")
    continuous = np.asarray(CONTINUOUS_FEATURE_INDICES, dtype=int)
    values[:, continuous] = scaler.transform(values[:, continuous]).astype(np.float32)
    padded = np.zeros((max_sequence_length, len(FEATURE_NAMES)), dtype=np.float32)
    padded[:len(values)] = values
    return padded


def _note_midi(note: str) -> int:
    match = NOTE_RE.fullmatch(note.strip())
    if match is None:
        raise LSTMFeatureError(f"Malformed melody note: {note!r}")
    letter, accidental, octave_text = match.groups()
    pitch_class = PITCH_CLASS_INDEX.get(letter + accidental)
    if pitch_class is None:
        raise LSTMFeatureError(f"Unsupported melody note: {note!r}")
    midi = (int(octave_text) + 1) * 12 + pitch_class
    if not 0 <= midi <= 127:
        raise LSTMFeatureError(f"MIDI pitch is outside 0–127: {note!r}")
    return midi


def extract_melody_sequence(row: Mapping[str, Any] | pd.Series) -> np.ndarray:
    """Return one `(notes, features)` sequence, with musical key context on each note."""
    try:
        notes = [part.strip() for part in str(row["melody_notes"]).split(",")]
        frequencies = np.asarray([float(part.strip()) for part in str(row["note_frequencies"]).split(",")], dtype=float)
        durations = np.asarray([float(part.strip()) for part in str(row["note_durations"]).split(",")], dtype=float)
        key_parts = str(row["musical_key"]).strip().split()
    except (KeyError, TypeError, ValueError) as error:
        raise LSTMFeatureError(f"Missing or malformed sequence fields: {error}") from error
    if not notes or any(not note for note in notes):
        raise LSTMFeatureError("Melody note list must be nonempty.")
    if not (len(notes) == len(frequencies) == len(durations)):
        raise LSTMFeatureError("Melody note, frequency, and duration list lengths must match.")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise LSTMFeatureError("Frequencies must be finite and positive.")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise LSTMFeatureError("Durations must be finite and positive.")
    if len(key_parts) != 2 or key_parts[0] not in PITCH_CLASS_INDEX or key_parts[1].lower() not in {"major", "minor"}:
        raise LSTMFeatureError(f"Malformed musical key: {row['musical_key']!r}")

    midi = np.asarray([_note_midi(note) for note in notes], dtype=np.int32)
    key_root = PITCH_CLASS_INDEX[key_parts[0]]
    key_is_minor = key_parts[1].lower() == "minor"
    key_root_one_hot = np.eye(12, dtype=np.float32)[key_root]
    rows: list[np.ndarray] = []
    for index, pitch in enumerate(midi):
        if index == 0:
            interval = 0
            repeated = ascending = descending = 0
        else:
            interval = int(pitch - midi[index - 1])
            repeated = int(interval == 0)
            ascending = int(interval > 0)
            descending = int(interval < 0)
        pitch_class = int(pitch % 12)
        relative_pc = int((pitch_class - key_root) % 12)
        row_features = np.concatenate((
            np.asarray([pitch, frequencies[index], pitch / 127.0, durations[index], interval, abs(interval)], dtype=np.float32),
            np.eye(12, dtype=np.float32)[pitch_class],
            np.eye(12, dtype=np.float32)[relative_pc],
            np.asarray([repeated, ascending, descending], dtype=np.float32),
            key_root_one_hot,
            np.asarray([float(key_is_minor)], dtype=np.float32),
        ))
        rows.append(row_features)
    sequence = np.vstack(rows).astype(np.float32, copy=False)
    if sequence.shape[1] != len(FEATURE_NAMES) or not np.isfinite(sequence).all():
        raise LSTMFeatureError("Sequence has an invalid feature count or non-finite values.")
    return sequence

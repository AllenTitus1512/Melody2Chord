"""Numerical melody features for the Random Forest baseline."""
from __future__ import annotations

from typing import Any

import librosa
import numpy as np
import pandas as pd
import re
from collections.abc import Mapping

PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FEATURE_NAMES = (
    *(f"pitch_class_{name}" for name in PITCH_CLASS_NAMES),
    "note_count", "mean_midi", "median_midi", "pitch_range", "mean_frequency_hz", "median_frequency_hz",
    "frequency_range_hz", "mean_note_duration", "total_note_duration", "mean_absolute_interval",
    "max_absolute_interval", *(f"key_root_{name}" for name in PITCH_CLASS_NAMES), "key_is_minor", "scale_note_ratio",
)

TRAINING_FEATURE_NAMES = (
    *(f"pitch_class_{name}" for name in ("C", "Csharp", "D", "Dsharp", "E", "F", "Fsharp", "G", "Gsharp", "A", "Asharp", "B")),
    "mean_frequency", "std_frequency", "min_frequency", "max_frequency",
    "mean_midi_pitch", "std_midi_pitch", "min_midi_pitch", "max_midi_pitch",
    "note_count", "total_melody_duration", "mean_note_duration", "std_note_duration", "min_note_duration", "max_note_duration",
    "mean_interval", "std_interval", "min_interval", "max_interval", "mean_absolute_interval",
    "ascending_interval_proportion", "descending_interval_proportion", "repeated_note_proportion",
    "key_root_pitch_class", "key_mode",
)
STRONGEST_PC_NAMES = ("C", "Csharp", "D", "Dsharp", "E", "F", "Fsharp", "G", "Gsharp", "A", "Asharp", "B")
V2_TRAINING_FEATURE_NAMES = (
    *TRAINING_FEATURE_NAMES,
    *(f"pitch_class_relative_to_key_{name}" for name in STRONGEST_PC_NAMES),
    "melody_range_semitones", "pitch_class_entropy",
    *(f"strongest_pitch_class_{name}" for name in STRONGEST_PC_NAMES),
    "note_density", "total_pitch_movement_semitones", "largest_upward_interval_semitones",
    "largest_downward_interval_semitones", "fraction_repeated_notes", "fraction_notes_in_key_scale",
)
_DATASET_NOTE_PATTERN = re.compile(r"^([A-G])([#b]?)(-?\d+)$")
_DATASET_PITCH_CLASSES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
                          "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


class FeatureExtractionError(ValueError):
    """Raised when melody records cannot be converted to reliable numerical features."""


def _parse_csv_numbers(value: Any, expected_count: int, default: float) -> list[float]:
    """Parse optional comma-separated values, with a documented default if omitted."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return [default] * expected_count
    try:
        numbers = [float(item.strip()) for item in str(value).split(",")]
    except ValueError as error:
        raise FeatureExtractionError("Frequency and duration columns must contain comma-separated numbers.") from error
    if len(numbers) != expected_count:
        raise FeatureExtractionError("Each optional frequency/duration list must match the melody note count.")
    return numbers


def _parse_key(key: str) -> tuple[int, bool]:
    """Convert a key such as 'G Major' or 'A Minor' to root and mode values."""
    parts = str(key).strip().split()
    if len(parts) != 2 or parts[0] not in PITCH_CLASS_NAMES or parts[1].lower() not in {"major", "minor"}:
        raise FeatureExtractionError("Musical keys must use names such as 'G Major' or 'A Minor'.")
    return PITCH_CLASS_NAMES.index(parts[0]), parts[1].lower() == "minor"


def _notes_to_midi(notes: list[str]) -> np.ndarray:
    try:
        values = np.array([librosa.note_to_midi(note.strip()) for note in notes], dtype=float)
    except (ValueError, TypeError) as error:
        raise FeatureExtractionError("Melody notes must be comma-separated names such as C4,E4,G4.") from error
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise FeatureExtractionError("Melody must contain at least one valid note.")
    return values


def _build_feature_vector(midi: np.ndarray, durations: np.ndarray, frequencies: np.ndarray, musical_key: str) -> np.ndarray:
    pitch_classes = np.mod(np.rint(midi).astype(int), 12)
    chroma = np.bincount(pitch_classes, weights=durations, minlength=12).astype(float)
    chroma /= chroma.sum()
    intervals = np.abs(np.diff(midi))
    key_root, is_minor = _parse_key(musical_key)
    key_vector = np.zeros(12)
    key_vector[key_root] = 1.0
    scale_intervals = (0, 2, 3, 5, 7, 8, 10) if is_minor else (0, 2, 4, 5, 7, 9, 11)
    scale_pitch_classes = {(key_root + interval) % 12 for interval in scale_intervals}
    scale_ratio = float(sum(pc in scale_pitch_classes for pc in pitch_classes) / len(pitch_classes))
    summary = np.array([
        len(midi), np.mean(midi), np.median(midi), np.ptp(midi), np.mean(frequencies), np.median(frequencies),
        np.ptp(frequencies), np.mean(durations), np.sum(durations), np.mean(intervals) if len(intervals) else 0.0,
        np.max(intervals) if len(intervals) else 0.0,
    ])
    return np.concatenate([chroma, summary, key_vector, [float(is_minor), scale_ratio]])


def extract_features(melody: dict[str, Any] | Mapping[str, Any] | pd.Series, musical_key: str | None = None) -> np.ndarray:
    """Extract a POP909 CSV row when called with one argument; retain the Step 2 API otherwise."""
    if musical_key is None and isinstance(melody, Mapping):
        return extract_dataset_features(melody)
    if musical_key is None and isinstance(melody, pd.Series):
        return extract_dataset_features(melody)
    assert musical_key is not None
    # Turn the Step 2 melody representation plus its detected key into one feature vector.
    timeline = melody.get("timeline", [])
    if not isinstance(timeline, list) or not timeline:
        raise FeatureExtractionError("Melody timeline is empty.")
    valid_items = [item for item in timeline if isinstance(item, dict) and item.get("note")]
    notes = [item["note"] for item in valid_items]
    if not notes:
        raise FeatureExtractionError("Melody timeline has no valid notes.")
    midi = _notes_to_midi(notes)
    durations = np.maximum(np.array([float(item.get("duration", 0.01)) for item in valid_items]), 0.01)
    derived_frequencies = librosa.midi_to_hz(midi)
    frequencies = np.array([float(item.get("frequency", derived)) for item, derived in zip(valid_items, derived_frequencies)])
    frequencies = np.where(np.isfinite(frequencies) & (frequencies > 0), frequencies, derived_frequencies)
    return _build_feature_vector(midi, durations, frequencies, musical_key)


def extract_dataset_features(row: Mapping[str, Any] | pd.Series) -> np.ndarray:
    """Convert one validated POP909 dataset row to deterministic fixed-length features."""
    try:
        notes = [part.strip() for part in str(row["melody_notes"]).split(",")]
        frequencies = np.asarray([float(part.strip()) for part in str(row["note_frequencies"]).split(",")], dtype=float)
        durations = np.asarray([float(part.strip()) for part in str(row["note_durations"]).split(",")], dtype=float)
        key = str(row["musical_key"]).strip().split()
    except (KeyError, TypeError, ValueError) as error:
        raise FeatureExtractionError(f"Missing or malformed melody feature fields: {error}") from error
    if not notes or any(not note for note in notes):
        raise FeatureExtractionError("Melody notes must be a nonempty comma-separated list.")
    if not (len(notes) == len(frequencies) == len(durations)):
        raise FeatureExtractionError("Note, frequency, and duration list lengths must match.")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise FeatureExtractionError("Frequencies must be finite and positive.")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise FeatureExtractionError("Durations must be finite and positive.")
    midi_values: list[int] = []
    for note in notes:
        match = _DATASET_NOTE_PATTERN.fullmatch(note)
        if match is None:
            raise FeatureExtractionError(f"Malformed melody note: {note!r}")
        letter, accidental, octave_text = match.groups()
        pitch_class = _DATASET_PITCH_CLASSES.get(letter + accidental)
        if pitch_class is None:
            raise FeatureExtractionError(f"Unsupported melody note: {note!r}")
        midi_values.append((int(octave_text) + 1) * 12 + pitch_class)
    if len(key) != 2 or key[0] not in _DATASET_PITCH_CLASSES or key[1].lower() not in {"major", "minor"}:
        raise FeatureExtractionError(f"Malformed musical key: {row['musical_key']!r}")
    midi = np.asarray(midi_values, dtype=float)
    intervals = np.diff(midi)
    pitch_classes = np.mod(midi.astype(int), 12)
    histogram = np.bincount(pitch_classes, weights=durations, minlength=12).astype(float)
    histogram /= histogram.sum()
    zeros = np.zeros(1, dtype=float)
    interval_values = intervals if intervals.size else zeros
    vector = np.asarray([
        *histogram,
        np.mean(frequencies), np.std(frequencies), np.min(frequencies), np.max(frequencies),
        np.mean(midi), np.std(midi), np.min(midi), np.max(midi),
        len(notes), np.sum(durations), np.mean(durations), np.std(durations), np.min(durations), np.max(durations),
        np.mean(interval_values), np.std(interval_values) if intervals.size else 0.0,
        np.min(interval_values), np.max(interval_values), np.mean(np.abs(intervals)) if intervals.size else 0.0,
        np.mean(intervals > 0) if intervals.size else 0.0,
        np.mean(intervals < 0) if intervals.size else 0.0,
        np.mean(intervals == 0) if intervals.size else 0.0,
        _DATASET_PITCH_CLASSES[key[0]], float(key[1].lower() == "minor"),
    ], dtype=float)
    if vector.size != len(TRAINING_FEATURE_NAMES) or not np.all(np.isfinite(vector)):
        raise FeatureExtractionError("Feature vector has an invalid length or non-finite value.")
    return vector


def extract_dataset_features_v2(row: Mapping[str, Any] | pd.Series) -> np.ndarray:
    """Add key-relative and melodic-shape features to all original dataset features.

    Minor-key scale membership uses the natural minor scale (Aeolian): 0, 2, 3,
    5, 7, 8, 10 semitones above the key root.
    """
    original = extract_dataset_features(row)
    note_names = [part.strip() for part in str(row["melody_notes"]).split(",")]
    durations = np.asarray([float(part.strip()) for part in str(row["note_durations"]).split(",")], dtype=float)
    key = str(row["musical_key"]).strip().split()
    key_root = _DATASET_PITCH_CLASSES[key[0]]
    is_minor = key[1].lower() == "minor"
    midi = np.asarray([
        (int(_DATASET_NOTE_PATTERN.fullmatch(note).group(3)) + 1) * 12
        + _DATASET_PITCH_CLASSES[_DATASET_NOTE_PATTERN.fullmatch(note).group(1) + _DATASET_NOTE_PATTERN.fullmatch(note).group(2)]
        for note in note_names
    ], dtype=int)
    pitch_classes = midi % 12
    histogram = original[:12]
    relative_histogram = np.zeros(12, dtype=float)
    np.add.at(relative_histogram, (pitch_classes - key_root) % 12, durations)
    relative_histogram /= relative_histogram.sum()
    entropy = float(-np.sum(histogram[histogram > 0] * np.log2(histogram[histogram > 0])))
    strongest = np.zeros(12, dtype=float)
    strongest[int(np.argmax(histogram))] = 1.0
    intervals = np.diff(midi)
    total_duration = float(np.sum(durations))
    scale_intervals = {0, 2, 3, 5, 7, 8, 10} if is_minor else {0, 2, 4, 5, 7, 9, 11}
    in_scale_fraction = float(np.mean([((pitch - key_root) % 12) in scale_intervals for pitch in pitch_classes]))
    extra = np.asarray([
        *relative_histogram,
        float(np.max(midi) - np.min(midi)), entropy,
        *strongest,
        len(note_names) / total_duration,
        float(np.sum(np.abs(intervals))),
        float(np.max(intervals[intervals > 0])) if np.any(intervals > 0) else 0.0,
        float(np.max(np.abs(intervals[intervals < 0]))) if np.any(intervals < 0) else 0.0,
        float(np.sum(intervals == 0) / len(note_names)),
        in_scale_fraction,
    ], dtype=float)
    vector = np.concatenate([original, extra])
    if vector.size != len(V2_TRAINING_FEATURE_NAMES) or not np.all(np.isfinite(vector)):
        raise FeatureExtractionError("V2 feature vector has an invalid length or non-finite value.")
    return vector


def dataframe_to_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build a Scikit-learn-ready feature matrix from validated CSV rows."""
    vectors: list[np.ndarray] = []
    row_errors: list[str] = []
    for index, row in dataframe.iterrows():
        notes = [note.strip() for note in str(row["melody_notes"]).split(",") if note.strip()]
        try:
            midi = _notes_to_midi(notes)
            durations = np.array(_parse_csv_numbers(row.get("note_durations"), len(notes), 1.0), dtype=float)
            raw_frequencies = np.array(_parse_csv_numbers(row.get("note_frequencies"), len(notes), np.nan), dtype=float)
            if np.any(durations <= 0):
                raise FeatureExtractionError("Note durations must be positive.")
            derived_frequencies = librosa.midi_to_hz(midi)
            frequencies = np.where(np.isfinite(raw_frequencies) & (raw_frequencies > 0), raw_frequencies, derived_frequencies)
            vectors.append(_build_feature_vector(midi, durations, frequencies, row["musical_key"]))
        except FeatureExtractionError as error:
            row_errors.append(f"row {index}: {error}")
    if row_errors:
        preview = "; ".join(row_errors[:3])
        raise FeatureExtractionError(f"Feature preparation failed for {len(row_errors)} row(s): {preview}")
    return pd.DataFrame(vectors, columns=FEATURE_NAMES, index=dataframe.index)

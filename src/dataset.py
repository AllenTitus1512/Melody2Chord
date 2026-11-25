"""Loading and validation for real melody-to-chord training data."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "training" / "melody_chord_dataset.csv"
COMMON_CHORD_CLASSES = ("C", "Dm", "Em", "F", "G", "Am", "A", "D", "E", "Bm", "G7", "C7")
REQUIRED_COLUMNS = ("melody_notes", "musical_key", "chord_label")
OPTIONAL_COLUMNS = ("note_frequencies", "note_durations")


class DatasetValidationError(ValueError):
    """Raised when the training dataset is absent or unsuitable for training."""


def load_training_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load CSV data from the documented training location."""
    path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    if not path.is_file():
        raise DatasetValidationError(
            f"Dataset not found: {path}. Place a real labeled CSV at "
            "data/training/melody_chord_dataset.csv; see data/training/README.md."
        )
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise DatasetValidationError(f"Could not read dataset '{path}': {error}") from error


def validate_training_dataset(
    dataframe: pd.DataFrame, allowed_chords: tuple[str, ...] = COMMON_CHORD_CLASSES
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate records, remove explicitly reported invalid rows, and return a report."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise DatasetValidationError(f"Dataset is missing required columns: {', '.join(missing_columns)}.")

    report: dict[str, object] = {"loaded": len(dataframe), "removed": 0, "duplicates": 0, "invalid": 0}
    prepared = dataframe.copy()
    for column in REQUIRED_COLUMNS:
        prepared[column] = prepared[column].fillna("").astype(str).str.strip()

    empty_mask = prepared["melody_notes"].eq("") | prepared["musical_key"].eq("") | prepared["chord_label"].eq("")
    invalid_chord_mask = ~prepared["chord_label"].isin(allowed_chords)
    invalid_mask = empty_mask | invalid_chord_mask
    report["invalid"] = int(invalid_mask.sum())
    prepared = prepared.loc[~invalid_mask].copy()

    duplicate_mask = prepared.duplicated()
    report["duplicates"] = int(duplicate_mask.sum())
    prepared = prepared.loc[~duplicate_mask].reset_index(drop=True)
    report["removed"] = int(report["invalid"]) + int(report["duplicates"])
    report["usable"] = len(prepared)
    report["class_counts"] = dict(Counter(prepared["chord_label"]))
    report["available_classes"] = sorted(prepared["chord_label"].unique().tolist())

    if prepared.empty:
        raise DatasetValidationError("No usable rows remain after dataset validation.")
    return prepared, report

"""Read-only validation for the real POP909-derived melody-to-chord CSV."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.dataset import COMMON_CHORD_CLASSES
except ModuleNotFoundError:
    from dataset import COMMON_CHORD_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "training" / "melody_chord_dataset.csv"
REQUIRED_COLUMNS = ("song_id", "melody_notes", "note_frequencies", "note_durations", "musical_key", "chord_label", "start_time", "end_time")
NOTE_PATTERN = re.compile(r"^[A-G](?:#)?-?\d+$")


def _numbers(value: object) -> tuple[list[float], list[str]]:
    """Parse a comma-separated numeric list while retaining useful error details."""
    if pd.isna(value) or str(value).strip() == "":
        return [], ["empty list"]
    fields = str(value).split(",")
    parsed: list[float] = []
    errors: list[str] = []
    for position, field in enumerate(fields, start=1):
        try:
            number = float(field.strip())
        except ValueError:
            errors.append(f"item {position} is not numeric")
            continue
        if not np.isfinite(number):
            errors.append(f"item {position} is non-finite")
        parsed.append(number)
    return parsed, errors


def validate_dataset(path: Path = DATASET_PATH) -> dict[str, object]:
    """Validate the CSV without changing it, returning row errors and category counts."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Run python src/prepare_pop909_dataset.py first.")
    dataframe = pd.read_csv(path, keep_default_na=True)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    invalid_rows: list[tuple[int, list[str]]] = []
    categories: dict[str, set[int]] = {
        "invalid duration rows": set(), "invalid frequency rows": set(),
        "length mismatch rows": set(), "missing value rows": set(),
        "unsupported chord rows": set(),
    }
    for index, row in dataframe.iterrows():
        row_number = int(index) + 2  # Include the CSV header when reporting line numbers.
        reasons: list[str] = []
        for column in REQUIRED_COLUMNS:
            if pd.isna(row[column]) or str(row[column]).strip() == "":
                reasons.append(f"missing value: {column}")
                categories["missing value rows"].add(row_number)

        raw_notes = "" if pd.isna(row["melody_notes"]) else str(row["melody_notes"])
        notes = [item.strip() for item in raw_notes.split(",") if item.strip()]
        if not notes:
            if "empty melody notes" not in reasons:
                reasons.append("empty melody notes")
        elif not all(NOTE_PATTERN.fullmatch(note) for note in notes):
            reasons.append("malformed melody note")

        frequencies, frequency_errors = _numbers(row["note_frequencies"])
        durations, duration_errors = _numbers(row["note_durations"])
        frequency_bad = bool(frequency_errors) or any(value <= 0 for value in frequencies)
        duration_bad = bool(duration_errors) or any(value <= 0 for value in durations)
        if frequency_bad:
            categories["invalid frequency rows"].add(row_number)
            reasons.append("invalid frequencies: " + ", ".join(frequency_errors or ["value <= 0"]))
        if duration_bad:
            categories["invalid duration rows"].add(row_number)
            reasons.append("invalid durations: " + ", ".join(duration_errors or ["value <= 0"]))
        if len(notes) != len(frequencies) or len(notes) != len(durations):
            categories["length mismatch rows"].add(row_number)
            reasons.append(f"list-length mismatch (notes={len(notes)}, frequencies={len(frequencies)}, durations={len(durations)})")

        chord = "" if pd.isna(row["chord_label"]) else str(row["chord_label"]).strip()
        if chord not in COMMON_CHORD_CLASSES:
            categories["unsupported chord rows"].add(row_number)
            reasons.append(f"unsupported chord label: {chord!r}")
        try:
            start, end = float(row["start_time"]), float(row["end_time"])
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                reasons.append("invalid segment timing")
        except (TypeError, ValueError):
            reasons.append("invalid segment timing")
        if reasons:
            invalid_rows.append((row_number, reasons))

    return {
        "rows": len(dataframe), "unique_songs": int(dataframe["song_id"].nunique()),
        "chord_distribution": dict(sorted(Counter(dataframe["chord_label"].dropna()).items())),
        "invalid_rows": invalid_rows, "categories": {name: len(rows) for name, rows in categories.items()},
    }


def print_report(report: dict[str, object]) -> None:
    """Print the read-only validation report."""
    print("POP909 MELODY-CHORD DATASET VALIDATION")
    print(f"Rows checked: {report['rows']}")
    print(f"Valid rows: {report['rows'] - len(report['invalid_rows'])}")
    print(f"Invalid rows: {len(report['invalid_rows'])}")
    print(f"Unique songs: {report['unique_songs']}")
    print("Error categories:")
    for category, count in report["categories"].items():
        print(f"{category.title()}: {count}")
    print("Chord distribution:")
    for chord, count in report["chord_distribution"].items():
        print(f"{chord}: {count}")
    for row_number, reasons in report["invalid_rows"][:20]:
        print(f"  CSV row {row_number}: {'; '.join(reasons)}")
    if len(report["invalid_rows"]) > 20:
        print(f"  ... {len(report['invalid_rows']) - 20} additional invalid row(s)")


if __name__ == "__main__":
    try:
        report = validate_dataset()
        print_report(report)
        if report["invalid_rows"]:
            raise SystemExit(1)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Validation stopped: {error}")
        raise SystemExit(1)

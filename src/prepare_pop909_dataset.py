"""Build real, time-aligned melody-to-chord rows from the official POP909 files.

This module intentionally performs dataset preparation only. It never creates
synthetic samples and never trains or saves a machine-learning model.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pretty_midi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POP909_ROOT = PROJECT_ROOT / "data" / "raw" / "POP909"
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "pop909_segments.csv"
TRAINING_CSV = PROJECT_ROOT / "data" / "training" / "melody_chord_dataset.csv"
STATISTICS_JSON = PROJECT_ROOT / "data" / "processed" / "pop909_dataset_statistics.json"
SUPPORTED_CHORDS = ("C", "Dm", "Em", "F", "G", "Am", "A", "D", "E", "Bm", "G7", "C7")
EPSILON = 1e-6
PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}
CSV_COLUMNS = ("song_id", "melody_notes", "note_frequencies", "note_durations", "musical_key", "chord_label", "start_time", "end_time")


class POP909PreparationError(RuntimeError):
    """Raised when official POP909 files are missing or cannot be processed."""


@dataclass(frozen=True)
class TimedLabel:
    """A label with start/end times from a POP909 whitespace-delimited annotation."""
    start: float
    end: float
    label: str


def _normalise_root(root: str) -> str | None:
    root = root.strip()
    if root in FLAT_TO_SHARP:
        return FLAT_TO_SHARP[root]
    return root if root in PITCH_CLASS_NAMES else None


def normalize_chord_label(label: str) -> str | None:
    """Map only exact, supported POP909 chord forms; skip extensions/inversions safely.

    Examples: ``C:maj`` -> ``C`` and ``D:min`` -> ``Dm``. Labels such as
    ``F#:maj7/5`` and ``Bb:min7`` are deliberately not simplified, because that
    would invent a target class that the annotation did not explicitly provide.
    """
    match = re.fullmatch(r"([A-G](?:#|b)?):(maj|min|7)", label.strip())
    if not match:
        return None
    root = _normalise_root(match.group(1))
    if root is None:
        return None
    quality = match.group(2)
    candidate = root if quality == "maj" else f"{root}m" if quality == "min" else f"{root}7"
    return candidate if candidate in SUPPORTED_CHORDS else None


def normalize_key_label(label: str) -> str | None:
    """Convert POP909 labels (for example ``Gb:maj``) to ``F# Major``."""
    match = re.fullmatch(r"([A-G](?:#|b)?):(maj|min)", label.strip())
    if not match:
        return None
    root = _normalise_root(match.group(1))
    if root is None:
        return None
    return f"{root} {'Major' if match.group(2) == 'maj' else 'Minor'}"


def read_timed_labels(path: Path) -> list[TimedLabel]:
    """Read POP909's three-column start/end/label annotation files."""
    labels: list[TimedLabel] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise POP909PreparationError(f"Could not read annotation '{path}': {error}") from error
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            start, end = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        labels.append(TimedLabel(start, end, fields[2]))
    return labels


def find_song_root(dataset_root: Path) -> Path:
    """Support both extracted ZIP and cloned-repository POP909 layouts."""
    nested = dataset_root / "POP909"
    return nested if nested.is_dir() else dataset_root


def discover_song_directories(dataset_root: Path) -> list[Path]:
    root = find_song_root(dataset_root)
    return sorted(path for path in root.iterdir() if path.is_dir() and re.fullmatch(r"\d{3}", path.name))


def find_melody_instrument(midi: pretty_midi.PrettyMIDI) -> pretty_midi.Instrument | None:
    """Find the official named MELODY MIDI instrument without guessing another part."""
    exact = [instrument for instrument in midi.instruments if instrument.name.strip().upper() == "MELODY"]
    if exact:
        return exact[0]
    containing = [instrument for instrument in midi.instruments if "MELODY" in instrument.name.upper()]
    return containing[0] if len(containing) == 1 else None


def midi_note_name(pitch: int) -> str:
    """Convert an integer MIDI pitch to a scientific note name, e.g. 60 -> C4."""
    return f"{PITCH_CLASS_NAMES[pitch % 12]}{pitch // 12 - 1}"


def melody_notes_for_interval(
    notes: Iterable[pretty_midi.Note], start: float, end: float
) -> list[tuple[str, float, float]]:
    """Return notes that genuinely overlap a chord interval using overlap durations."""
    selected: list[tuple[str, float, float]] = []
    for note in notes:
        overlap_start = max(float(note.start), start)
        overlap_end = min(float(note.end), end)
        overlap_duration = overlap_end - overlap_start
        if note.pitch < 0 or note.pitch > 127 or not np.isfinite(overlap_duration) or overlap_duration <= EPSILON:
            continue
        frequency = 440.0 * 2 ** ((note.pitch - 69) / 12)
        if not np.isfinite(frequency) or frequency <= 0:
            continue
        selected.append((midi_note_name(note.pitch), frequency, overlap_duration))
    return selected


def key_for_interval(key_labels: list[TimedLabel], start: float, end: float) -> str | None:
    """Choose the valid key label having the greatest temporal overlap with a chord."""
    matches: list[tuple[float, str]] = []
    for key_label in key_labels:
        overlap = min(end, key_label.end) - max(start, key_label.start)
        key = normalize_key_label(key_label.label)
        if overlap > 0 and key is not None:
            matches.append((overlap, key))
    return max(matches, default=(0.0, None), key=lambda item: item[0])[1]


def _song_files(song_directory: Path) -> tuple[Path, Path, Path]:
    song_id = song_directory.name
    midi_path = song_directory / f"{song_id}.mid"
    # chord_midi is used because it shares the MIDI timebase with the MELODY track.
    chord_path = song_directory / "chord_midi.txt"
    key_path = song_directory / "key_audio.txt"
    missing = [str(path.name) for path in (midi_path, chord_path, key_path) if not path.is_file()]
    if missing:
        raise POP909PreparationError(f"Missing required file(s) for song {song_id}: {', '.join(missing)}")
    return midi_path, chord_path, key_path


def process_song(song_directory: Path, statistics: dict[str, object]) -> list[dict[str, object]]:
    """Create genuine training rows by aligning the official melody and annotations."""
    midi_path, chord_path, key_path = _song_files(song_directory)
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as error:
        raise POP909PreparationError(f"Could not parse MIDI for song {song_directory.name}: {error}") from error
    melody_instrument = find_melody_instrument(midi)
    if melody_instrument is None:
        raise POP909PreparationError(f"No MELODY track found in song {song_directory.name}.")
    chord_labels = read_timed_labels(chord_path)
    key_labels = read_timed_labels(key_path)
    if not chord_labels:
        raise POP909PreparationError(f"No chord annotations found for song {song_directory.name}.")
    if not key_labels:
        raise POP909PreparationError(f"No key annotations found for song {song_directory.name}.")

    rows: list[dict[str, object]] = []
    notes = sorted(melody_instrument.notes, key=lambda item: (item.start, item.end, item.pitch))
    for chord in chord_labels:
        statistics["total_chord_segments"] = int(statistics["total_chord_segments"]) + 1
        normalized_chord = normalize_chord_label(chord.label)
        if normalized_chord is None:
            unsupported = statistics["unsupported_chord_labels"]
            assert isinstance(unsupported, Counter)
            unsupported[chord.label] += 1
            statistics["skipped_segments"] = int(statistics["skipped_segments"]) + 1
            continue
        musical_key = key_for_interval(key_labels, chord.start, chord.end)
        overlapping_notes = melody_notes_for_interval(notes, chord.start, chord.end)
        if musical_key is None or not overlapping_notes:
            statistics["skipped_segments"] = int(statistics["skipped_segments"]) + 1
            continue
        note_names, frequencies, durations = zip(*overlapping_notes)
        if not note_names or not (len(note_names) == len(frequencies) == len(durations)):
            statistics["skipped_segments"] = int(statistics["skipped_segments"]) + 1
            continue
        if any(not np.isfinite(frequency) or frequency <= 0 for frequency in frequencies):
            statistics["skipped_segments"] = int(statistics["skipped_segments"]) + 1
            continue
        if any(not np.isfinite(duration) or duration <= EPSILON for duration in durations):
            statistics["skipped_segments"] = int(statistics["skipped_segments"]) + 1
            continue
        rows.append({
            "song_id": song_directory.name,
            "melody_notes": ",".join(note_names),
            "note_frequencies": ",".join(f"{frequency:.6f}" for frequency in frequencies),
            "note_durations": ",".join(f"{duration:.6f}" for duration in durations),
            "musical_key": musical_key,
            "chord_label": normalized_chord,
            "start_time": round(chord.start, 6),
            "end_time": round(chord.end, 6),
        })
    return rows


def prepare_pop909_dataset(dataset_root: Path = POP909_ROOT) -> dict[str, object]:
    """Generate CSV and JSON statistics exclusively from a locally supplied POP909 copy."""
    if not dataset_root.is_dir() or not any(dataset_root.iterdir()):
        raise POP909PreparationError(
            "POP909 dataset not found. Expected location:\n"
            f"{dataset_root}\n"
            "Download the official POP909 repository/ZIP from "
            "https://github.com/music-x-lab/POP909-Dataset and place its POP909 song folders here."
        )
    song_directories = discover_song_directories(dataset_root)
    if not song_directories:
        raise POP909PreparationError("No POP909 numbered song folders or MIDI files were found.")

    statistics: dict[str, object] = {
        "dataset_name": "POP909", "songs_discovered": len(song_directories), "songs_processed": 0,
        "songs_failed": 0, "total_chord_segments": 0, "valid_segments": 0, "skipped_segments": 0,
        "duplicate_rows_removed": 0, "unsupported_chord_labels": Counter(), "song_failures": [],
        "successfully_parsed_song_ids": [],
    }
    rows: list[dict[str, object]] = []
    for song_directory in song_directories:
        try:
            song_rows = process_song(song_directory, statistics)
            rows.extend(song_rows)
            statistics["songs_processed"] = int(statistics["songs_processed"]) + 1
            statistics["successfully_parsed_song_ids"].append(song_directory.name)
        except POP909PreparationError as error:
            statistics["songs_failed"] = int(statistics["songs_failed"]) + 1
            failures = statistics["song_failures"]
            assert isinstance(failures, list)
            failures.append(str(error))
            print(f"Warning: {error}")

    dataframe = pd.DataFrame(rows, columns=CSV_COLUMNS)
    if dataframe.empty:
        raise POP909PreparationError("No valid melody/chord segments were generated; no CSV was written.")
    before_duplicates = len(dataframe)
    dataframe = dataframe.drop_duplicates().reset_index(drop=True)
    statistics["duplicate_rows_removed"] = before_duplicates - len(dataframe)
    statistics["valid_segments"] = len(dataframe)
    statistics["unique_chord_classes"] = int(dataframe["chord_label"].nunique())
    statistics["examples_per_chord_class"] = dict(sorted(Counter(dataframe["chord_label"]).items()))
    statistics["unique_songs"] = int(dataframe["song_id"].nunique())
    represented_songs = set(dataframe["song_id"].astype(str))
    parsed_songs = set(statistics["successfully_parsed_song_ids"])
    statistics["songs_successfully_parsed"] = len(parsed_songs)
    statistics["songs_represented"] = len(represented_songs)
    statistics["songs_with_zero_accepted_segments"] = sorted(parsed_songs - represented_songs)
    segment_durations = dataframe["end_time"] - dataframe["start_time"]
    statistics["minimum_segment_duration"] = round(float(segment_durations.min()), 6)
    statistics["maximum_segment_duration"] = round(float(segment_durations.max()), 6)
    statistics["average_segment_duration"] = round(float(segment_durations.mean()), 6)
    statistics["unsupported_chord_labels"] = dict(sorted(statistics["unsupported_chord_labels"].items()))
    statistics.pop("successfully_parsed_song_ids")

    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(PROCESSED_CSV, index=False)
    dataframe.to_csv(TRAINING_CSV, index=False)
    STATISTICS_JSON.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    return statistics


def print_statistics(statistics: dict[str, object]) -> None:
    """Print a concise processing report after successful preparation."""
    print("\nPOP909 DATASET PREPARATION COMPLETE")
    print(f"Songs discovered: {statistics['songs_discovered']}")
    print(f"Songs processed: {statistics['songs_processed']}")
    print(f"Songs successfully parsed: {statistics['songs_successfully_parsed']}")
    print(f"Songs represented in final training CSV: {statistics['songs_represented']}")
    zero_segment_songs = statistics["songs_with_zero_accepted_segments"]
    print(f"Songs with zero accepted training segments: {len(zero_segment_songs)}")
    print(f"Song IDs with zero accepted segments: {', '.join(zero_segment_songs) if zero_segment_songs else 'None'}")
    print(f"Songs failed: {statistics['songs_failed']}")
    print(f"Valid melody/chord segments: {statistics['valid_segments']}")
    print(f"Skipped segments: {statistics['skipped_segments']}")
    print(f"Duplicate rows removed: {statistics['duplicate_rows_removed']}")
    print("\nChord distribution:")
    for chord, count in statistics["examples_per_chord_class"].items():
        print(f"{chord}: {count}")
    print("\nUnsupported chord labels:")
    unsupported = statistics["unsupported_chord_labels"]
    if unsupported:
        for label, count in unsupported.items():
            print(f"{label}: {count}")
    else:
        print("None")
    print(f"\nOutput: {TRAINING_CSV}")


if __name__ == "__main__":
    try:
        print("Inspecting POP909 and preparing real melody-to-chord segments...")
        print_statistics(prepare_pop909_dataset())
    except POP909PreparationError as error:
        print(f"Dataset preparation stopped: {error}")
        raise SystemExit(1)


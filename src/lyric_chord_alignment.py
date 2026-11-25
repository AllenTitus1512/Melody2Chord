"""Timestamp-based alignment of transcribed lyric words and predicted chords."""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
MODEL_METADATA_PATH = ROOT / "models" / "chord_lstm_metadata.json"
DEFAULT_MAX_LINE_CHARS = 56
TIME_EPSILON = 1e-6
SENTENCE_END_RE = re.compile(r"[.!?]+[\"'’”)]*$")


class LyricChordAlignmentError(ValueError):
    """Raised when chord class metadata needed for safe alignment is unavailable."""


@lru_cache(maxsize=1)
def trained_chord_classes() -> tuple[str, ...]:
    """Read the trained label vocabulary without initializing TensorFlow."""
    try:
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        classes = metadata.get("chord_classes")
        if not isinstance(classes, list) or not classes or not all(isinstance(label, str) for label in classes):
            raise ValueError("chord_classes is missing or malformed")
        return tuple(classes)
    except Exception as error:
        raise LyricChordAlignmentError("The trained chord class list is unavailable for alignment.") from error


def _timestamp(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _word_ends_sentence(word: str) -> bool:
    return bool(SENTENCE_END_RE.search(word.rstrip()))


def _wrap_words(words: list[dict[str, Any]], max_line_chars: int) -> list[list[dict[str, Any]]]:
    """Wrap at punctuation or a readable character limit, keeping each word intact."""
    lines: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for word in words:
        text = word["word"]
        needed = len(text) + (1 if current else 0)
        if current and current_chars + needed > max_line_chars:
            lines.append(current)
            current, current_chars = [], 0
        current.append(word)
        current_chars += len(text) + (1 if len(current) > 1 else 0)
        if _word_ends_sentence(text):
            lines.append(current)
            current, current_chars = [], 0
    if current:
        lines.append(current)
    return lines


def align_lyrics_and_chords(
    words: Iterable[dict[str, Any]],
    chord_predictions: Iterable[dict[str, Any]],
    *,
    chord_classes: Iterable[str] | None = None,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
) -> dict[str, Any]:
    """Return readable lyric lines with chord transitions placed over timed words.

    Word order remains as transcribed. Invalid word or prediction times are skipped
    and described in `warnings`. A chord is placed only when its interval overlaps
    at least one word; among those words, the closest word start to the chord start
    is selected. Consecutive same-label, touching/overlapping predictions are
    suppressed in the sheet, while every valid original interval stays in
    `prediction_details` for inspection.
    """
    if not isinstance(max_line_chars, int) or max_line_chars < 12:
        raise ValueError("max_line_chars must be an integer of at least 12.")
    valid_classes = set(chord_classes if chord_classes is not None else trained_chord_classes())
    warnings: list[str] = []
    valid_words: list[dict[str, Any]] = []
    invalid_words = 0
    previous_word_start: float | None = None
    for original_index, item in enumerate(words or []):
        if not isinstance(item, dict):
            invalid_words += 1
            continue
        text = item.get("word")
        start, end = _timestamp(item.get("start")), _timestamp(item.get("end"))
        if not isinstance(text, str) or not text.strip() or start is None or end is None or start < 0 or end <= start:
            invalid_words += 1
            continue
        if previous_word_start is not None and start < previous_word_start:
            warnings.append("Word timestamps are not chronological; transcription order was preserved.")
        previous_word_start = start
        valid_words.append({"word": text.strip(), "start": start, "end": end,
                            "source_index": original_index})
    if invalid_words:
        warnings.append(f"Skipped {invalid_words} word(s) with missing or invalid text/timestamps.")

    valid_predictions: list[dict[str, Any]] = []
    invalid_predictions = 0
    unsupported_predictions = 0
    for item in chord_predictions or []:
        if not isinstance(item, dict):
            invalid_predictions += 1
            continue
        chord = item.get("chord")
        start, end = _timestamp(item.get("start")), _timestamp(item.get("end"))
        probability = _timestamp(item.get("probability")) if item.get("probability") is not None else None
        if not isinstance(chord, str) or start is None or end is None or start < 0 or end <= start:
            invalid_predictions += 1
            continue
        if probability is not None and not 0 <= probability <= 1:
            invalid_predictions += 1
            continue
        if chord not in valid_classes:
            unsupported_predictions += 1
            continue
        valid_predictions.append({"chord": chord, "start": start, "end": end,
                                  "probability": probability})
    valid_predictions.sort(key=lambda item: (item["start"], item["end"]))
    if invalid_predictions:
        warnings.append(f"Skipped {invalid_predictions} chord prediction(s) with invalid timestamps or probability.")
    if unsupported_predictions:
        warnings.append(f"Skipped {unsupported_predictions} chord prediction(s) outside the trained class list.")
    if not valid_predictions and chord_predictions:
        warnings.append("No valid trained-class chord suggestions were available to align.")

    # A pair of index maps allows line wrapping after alignment without losing word identity.
    word_to_chords: dict[int, list[dict[str, Any]]] = {}
    prediction_details: list[dict[str, Any]] = []
    unmatched_predictions = 0
    redundant_predictions = 0
    previous_emitted: dict[str, Any] | None = None
    for prediction_index, prediction in enumerate(valid_predictions):
        overlapping = [
            (index, word) for index, word in enumerate(valid_words)
            if min(prediction["end"], word["end"]) - max(prediction["start"], word["start"]) > TIME_EPSILON
        ]
        if not overlapping:
            unmatched_predictions += 1
            prediction_details.append({**prediction, "word_index": None, "aligned": False})
            continue
        word_index, _word = min(overlapping, key=lambda pair: (abs(pair[1]["start"] - prediction["start"]), pair[0]))
        detail = {**prediction, "word_index": word_index, "aligned": True}
        prediction_details.append(detail)
        adjacent_same = (
            previous_emitted is not None
            and previous_emitted["chord"] == prediction["chord"]
            and prediction["start"] <= previous_emitted["end"] + 0.1
        )
        if adjacent_same:
            redundant_predictions += 1
            # Extend the displayed event while retaining per-window detail above.
            previous_emitted["event"]["end"] = max(previous_emitted["event"]["end"], prediction["end"])
            previous_emitted["end"] = max(previous_emitted["end"], prediction["end"])
            continue
        event = {"chord": prediction["chord"], "word_index": word_index,
                 "time": prediction["start"], "start": prediction["start"],
                 "end": prediction["end"], "probability": prediction["probability"]}
        word_to_chords.setdefault(word_index, []).append(event)
        previous_emitted = {"chord": prediction["chord"], "end": prediction["end"],
                             "word_index": word_index, "prediction_index": prediction_index,
                             "event": event}
    if unmatched_predictions:
        warnings.append(f"{unmatched_predictions} chord prediction(s) did not overlap any timed lyric word and were not placed.")
    if redundant_predictions:
        warnings.append(f"Suppressed {redundant_predictions} adjacent repeated chord label(s) in the sheet; original intervals remain in prediction details.")
    if not valid_words:
        warnings.append("No valid timed lyric words are available to display.")

    wrapped = _wrap_words(valid_words, max_line_chars)
    lines: list[dict[str, Any]] = []
    global_index = 0
    for line_words in wrapped:
        line_start = global_index
        line_end = line_start + len(line_words)
        line_chords = []
        for word_index in range(line_start, line_end):
            for event in word_to_chords.get(word_index, []):
                line_chords.append({**event, "word_index": word_index - line_start,
                                    "global_word_index": word_index})
        lines.append({"words": [{"word": word["word"], "start": word["start"], "end": word["end"]}
                                for word in line_words],
                      "chords": sorted(line_chords, key=lambda chord: (chord["time"], chord["word_index"]))})
        global_index = line_end

    return {
        "lines": lines,
        "prediction_details": prediction_details,
        "warnings": list(dict.fromkeys(warnings)),
        "stats": {"valid_word_count": len(valid_words), "invalid_word_count": invalid_words,
                  "valid_prediction_count": len(valid_predictions),
                  "unmatched_prediction_count": unmatched_predictions,
                  "suppressed_repeated_chord_count": redundant_predictions},
    }

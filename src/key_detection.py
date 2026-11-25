"""Key detection from Step 2's extracted melody timeline."""
from __future__ import annotations

from typing import Any

import librosa
import numpy as np

# Krumhansl-Schmuckler tonal profiles, ordered C through B.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MINIMUM_NOTE_SEGMENTS = 3


class KeyDetectionError(Exception):
    """Raised when a melody does not contain enough usable tonal evidence."""


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate a stable Pearson correlation for two 12-element chroma vectors."""
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _scale_notes(root: int, mode: str) -> list[str]:
    """Return chromatic-name scale notes for the detected major or natural minor key."""
    intervals = (0, 2, 4, 5, 7, 9, 11) if mode == "Major" else (0, 2, 3, 5, 7, 8, 10)
    return [PITCH_CLASS_NAMES[(root + interval) % 12] for interval in intervals]


def _melody_chroma(melody: dict[str, Any]) -> np.ndarray:
    """Build a duration-weighted pitch-class histogram from the melody timeline."""
    timeline = melody.get("timeline")
    if not isinstance(timeline, list) or len(timeline) < MINIMUM_NOTE_SEGMENTS:
        raise KeyDetectionError("At least three reliable detected notes are needed to determine a musical key.")

    chroma = np.zeros(12, dtype=float)
    usable_segments = 0
    for item in timeline:
        if not isinstance(item, dict) or not isinstance(item.get("note"), str):
            continue
        try:
            pitch_class = int(round(librosa.note_to_midi(item["note"]))) % 12
            duration = float(item.get("duration", 0))
        except (ValueError, TypeError):
            continue
        # Duration gives sustained vocal notes an appropriately stronger tonal influence.
        chroma[pitch_class] += max(duration, 0.01)
        usable_segments += 1

    if usable_segments < MINIMUM_NOTE_SEGMENTS or chroma.sum() == 0:
        raise KeyDetectionError("The extracted melody does not contain enough reliable notes for key detection.")
    return chroma / chroma.sum()


def detect_key(melody: dict[str, Any]) -> dict[str, object]:
    """Identify the best-fitting major/minor key for a real extracted melody.

    The 12 pitch-class weights are compared against every rotated major and minor
    Krumhansl-Schmuckler profile. No chord or machine-learning prediction occurs here.
    """
    chroma = _melody_chroma(melody)
    candidates: list[tuple[float, int, str]] = []
    for root in range(12):
        candidates.append((_correlation(chroma, np.roll(MAJOR_PROFILE, root)), root, "Major"))
        candidates.append((_correlation(chroma, np.roll(MINOR_PROFILE, root)), root, "Minor"))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    best_score, root, mode = candidates[0]
    second_score = candidates[1][0]
    if best_score <= 0:
        raise KeyDetectionError("The melody does not provide a reliable tonal match. Try a longer, clearer vocal phrase.")

    # A transparent heuristic: profile fit plus separation from the next-best key.
    fit_strength = max(0.0, best_score)
    separation = max(0.0, best_score - second_score)
    confidence = round(min(100.0, (0.75 * fit_strength + 0.25 * separation) * 100), 1)
    return {
        "key": f"{PITCH_CLASS_NAMES[root]} {mode}",
        "confidence": confidence,
        "scale_notes": _scale_notes(root, mode),
    }

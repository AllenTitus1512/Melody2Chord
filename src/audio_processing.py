"""Audio utilities, including vocal melody extraction, independent from Flask."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import median_filter

# Exposed constants make it easy to tune the detector for different vocal ranges.
PYIN_FMIN_HZ = 80.0
PYIN_FMAX_HZ = 1000.0
VOICED_PROBABILITY_THRESHOLD = 0.55
HOP_LENGTH = 512
FRAME_LENGTH = 2048
SMOOTHING_WINDOW_FRAMES = 5
MIN_NOTE_DURATION_SECONDS = 0.12
MIN_AUDIO_DURATION_SECONDS = 0.25


class AudioProcessingError(Exception):
    """Raised when an audio file cannot provide usable processing results."""


def load_and_normalize_audio(file_path: str | Path) -> tuple[np.ndarray, int]:
    """Load a recording in mono and peak-normalize it without changing its rate."""
    try:
        audio, sampling_rate = librosa.load(str(file_path), sr=None, mono=True)
    except Exception as error:
        raise AudioProcessingError(
            "Unable to read this audio file. Try a valid WAV or MP3 recording."
        ) from error

    if audio.size == 0 or sampling_rate <= 0:
        raise AudioProcessingError("The uploaded audio file is empty or invalid.")

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio, int(sampling_rate)


def get_audio_info(file_path: str | Path) -> dict[str, float | int]:
    """Return basic metadata used by the upload interface."""
    audio, sampling_rate = load_and_normalize_audio(file_path)
    return {"sampling_rate": sampling_rate, "duration": round(len(audio) / sampling_rate, 3)}


def _smooth_midi_frames(midi_frames: np.ndarray) -> np.ndarray:
    """Median-smooth each voiced run, never smoothing pitch across a silence gap."""
    smoothed = midi_frames.copy()
    valid = ~np.isnan(midi_frames)
    index = 0
    while index < len(midi_frames):
        if not valid[index]:
            index += 1
            continue
        end = index
        while end < len(midi_frames) and valid[end]:
            end += 1
        run = midi_frames[index:end]
        if len(run) >= SMOOTHING_WINDOW_FRAMES:
            smoothed[index:end] = median_filter(run, size=SMOOTHING_WINDOW_FRAMES, mode="nearest")
        index = end
    return smoothed


def _build_note_segments(
    midi_frames: np.ndarray, frequencies: np.ndarray, sampling_rate: int, audio_duration: float
) -> list[dict[str, float | int]]:
    """Group equal, adjacent note frames into timestamped raw note segments."""
    segments: list[dict[str, float | int]] = []
    frame_duration = HOP_LENGTH / sampling_rate
    start = 0
    while start < len(midi_frames):
        if np.isnan(midi_frames[start]):
            start += 1
            continue
        midi = int(midi_frames[start])
        end = start + 1
        while end < len(midi_frames) and not np.isnan(midi_frames[end]) and int(midi_frames[end]) == midi:
            end += 1
        start_time = librosa.frames_to_time(start, sr=sampling_rate, hop_length=HOP_LENGTH)
        end_time = min(audio_duration, librosa.frames_to_time(end, sr=sampling_rate, hop_length=HOP_LENGTH))
        run_frequencies = frequencies[start:end]
        voiced_frequencies = run_frequencies[np.isfinite(run_frequencies)]
        segments.append({
            "start": float(start_time), "end": float(end_time), "midi": midi,
            "frequency": float(np.median(voiced_frequencies)), "frame_start": start, "frame_end": end,
        })
        start = end
    return segments


def _merge_short_segments(segments: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    """Absorb brief, adjacent pitch blips into their longer neighbouring note."""
    if len(segments) < 2:
        return segments
    index = 0
    while index < len(segments):
        segment = segments[index]
        duration = float(segment["end"]) - float(segment["start"])
        if duration >= MIN_NOTE_DURATION_SECONDS:
            index += 1
            continue
        previous = segments[index - 1] if index > 0 else None
        following = segments[index + 1] if index + 1 < len(segments) else None
        previous_adjacent = previous and int(previous["frame_end"]) == int(segment["frame_start"])
        following_adjacent = following and int(segment["frame_end"]) == int(following["frame_start"])
        target = previous if previous_adjacent else following if following_adjacent else None
        if target is None:
            index += 1
            continue
        # Keep the neighbouring pitch; the short segment is a likely detection fluctuation.
        if target is previous:
            previous["end"] = segment["end"]
            previous["frame_end"] = segment["frame_end"]
        else:
            following["start"] = segment["start"]
            following["frame_start"] = segment["frame_start"]
        segments.pop(index)
        if target is previous:
            index = max(0, index - 1)
    return segments


def extract_melody(audio_path: str | Path) -> dict[str, object]:
    """Extract a smoothed, voiced-note melody timeline from an uploaded recording.

    This function intentionally performs no key detection, chord generation, or ML inference.
    """
    audio, sampling_rate = load_and_normalize_audio(audio_path)
    audio_duration = len(audio) / sampling_rate
    if audio_duration < MIN_AUDIO_DURATION_SECONDS:
        raise AudioProcessingError("The recording is too short for reliable melody extraction.")

    try:
        frequencies, voiced_flags, voiced_probabilities = librosa.pyin(
            audio, fmin=PYIN_FMIN_HZ, fmax=PYIN_FMAX_HZ, sr=sampling_rate,
            hop_length=HOP_LENGTH, frame_length=FRAME_LENGTH,
        )
    except Exception as error:
        raise AudioProcessingError("Pitch detection failed for this recording.") from error

    if frequencies is None or voiced_flags is None or voiced_probabilities is None:
        raise AudioProcessingError("No detectable vocal melody was found in this recording.")

    reliable = (
        voiced_flags.astype(bool)
        & (voiced_probabilities >= VOICED_PROBABILITY_THRESHOLD)
        & np.isfinite(frequencies)
    )
    if not np.any(reliable):
        raise AudioProcessingError("No confident voiced melody was detected. Try a clearer vocal recording.")

    midi_frames = np.full(len(frequencies), np.nan)
    midi_frames[reliable] = np.rint(librosa.hz_to_midi(frequencies[reliable]))
    smoothed_midi = _smooth_midi_frames(midi_frames)
    segments = _merge_short_segments(_build_note_segments(smoothed_midi, frequencies, sampling_rate, audio_duration))

    timeline = [{
        "start": round(float(segment["start"]), 3),
        "end": round(float(segment["end"]), 3),
        "duration": round(float(segment["end"]) - float(segment["start"]), 3),
        "note": librosa.midi_to_note(int(segment["midi"]), unicode=False),
        "frequency": round(float(segment["frequency"]), 2),
    } for segment in segments]
    if not timeline:
        raise AudioProcessingError("No stable vocal notes could be extracted from this recording.")

    return {
        "notes": [item["note"] for item in timeline],
        "note_count": len(timeline),
        "melody_duration": round(sum(item["duration"] for item in timeline), 3),
        "timeline": timeline,
    }

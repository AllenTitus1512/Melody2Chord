"""Lazy faster-whisper transcription with validated, audio-relative word times."""
from __future__ import annotations

import logging
import math
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_MODEL_SIZE = "base"
MODEL_SIZE_ENV = "MELODY2CHORD_WHISPER_MODEL"
MAX_AUDIO_BYTES = 50 * 1024 * 1024
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3"}
LOGGER = logging.getLogger(__name__)


class LyricsTranscriptionError(Exception):
    """Safe-to-display failure with an HTTP status suitable for the API."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def _log_whisper_cache_diagnostics() -> None:
    """Log cache availability and free space after a model initialization failure."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_path = Path(HF_HUB_CACHE)
        existing_parent = cache_path
        while not existing_parent.exists() and existing_parent != existing_parent.parent:
            existing_parent = existing_parent.parent
        free_bytes = shutil.disk_usage(existing_parent).free
        LOGGER.error(
            "Whisper cache diagnostics: cache_dir=%s exists=%s writable=%s free_bytes=%s",
            cache_path,
            cache_path.is_dir(),
            os.access(cache_path if cache_path.exists() else existing_parent, os.W_OK),
            free_bytes,
        )
    except Exception:
        LOGGER.exception("Could not inspect the faster-whisper model cache")


@lru_cache(maxsize=2)
def _load_model(model_size: str) -> Any:
    """Initialize a local CPU model only when the first transcription is requested."""
    try:
        from faster_whisper import WhisperModel
        return WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as error:
        LOGGER.exception(
            "Could not initialize faster-whisper model '%s' (exception=%s: %s)",
            model_size,
            type(error).__name__,
            error,
        )
        _log_whisper_cache_diagnostics()
        raise LyricsTranscriptionError(
            "The lyrics transcription model could not be initialized. On first use it may need an internet connection to download the model.",
            503,
        ) from error


def clear_model_cache() -> None:
    """Clear initialized model instances (useful for tests and configuration changes)."""
    _load_model.cache_clear()


def _validate_audio_file(audio_path: str | Path) -> Path:
    path = Path(audio_path)
    if not path.is_file():
        raise LyricsTranscriptionError("The uploaded audio file was not found. Please upload it again.", 404)
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise LyricsTranscriptionError("Only WAV and MP3 audio files are supported.", 400)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise LyricsTranscriptionError("The uploaded audio file could not be accessed.", 422) from error
    if size == 0:
        raise LyricsTranscriptionError("The uploaded audio file is empty.", 400)
    if size > MAX_AUDIO_BYTES:
        raise LyricsTranscriptionError("The audio file is too large. Please use a file below 50 MB.", 413)
    return path


def _valid_language(language: str | None) -> str | None:
    if language is None or (isinstance(language, str) and not language.strip()):
        return None
    if not isinstance(language, str) or len(language.strip()) > 16:
        raise LyricsTranscriptionError("Language must be a short language code such as 'en', or omitted for automatic detection.", 400)
    return language.strip()


def transcribe_lyrics(audio_path: str | Path, language: str | None = None) -> dict[str, Any]:
    """Transcribe vocals and return actual word timestamps in seconds from audio start.

    Faster-whisper words without finite, positive-duration timestamps are skipped;
    the skip count is returned in metadata. No timing is inferred from transcript text.
    VAD is disabled because its speech assumptions can omit quiet or sustained singing.
    """
    path = _validate_audio_file(audio_path)
    language = _valid_language(language)
    model_size = os.getenv(MODEL_SIZE_ENV, DEFAULT_MODEL_SIZE).strip() or DEFAULT_MODEL_SIZE
    model = _load_model(model_size)
    try:
        segment_iterator, info = model.transcribe(
            str(path),
            language=language,
            task="transcribe",
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=True,
            word_timestamps=True,
            vad_filter=False,
        )
        # Transcription is lazy in faster-whisper; iterate to execute inference.
        segments = list(segment_iterator)
    except Exception as error:
        LOGGER.exception("faster-whisper transcription failed")
        raise LyricsTranscriptionError(
            "The audio could not be transcribed. Try a clear vocal recording in a supported format.", 422
        ) from error

    transcript = "".join(str(getattr(segment, "text", "") or "") for segment in segments).strip()
    if not transcript:
        raise LyricsTranscriptionError("No lyrics were recognized in this recording. Try a clearer vocal passage.", 422)

    try:
        duration = float(info.duration)
        if not math.isfinite(duration) or duration < 0:
            duration = None
    except (AttributeError, TypeError, ValueError):
        duration = None

    words: list[dict[str, Any]] = []
    skipped_word_timestamps = 0
    for segment in segments:
        for word_item in getattr(segment, "words", None) or []:
            word = str(getattr(word_item, "word", "") or "").strip()
            try:
                start = float(word_item.start)
                end = float(word_item.end)
            except (AttributeError, TypeError, ValueError):
                skipped_word_timestamps += 1
                continue
            if (not word or not math.isfinite(start) or not math.isfinite(end) or start < 0
                    or end <= start or (duration is not None and end > duration + 1e-6)):
                skipped_word_timestamps += 1
                continue
            result: dict[str, Any] = {"word": word, "start": start, "end": end}
            probability = getattr(word_item, "probability", None)
            try:
                probability = float(probability)
            except (TypeError, ValueError):
                probability = None
            if probability is not None and math.isfinite(probability) and 0.0 <= probability <= 1.0:
                result["probability"] = probability
            words.append(result)

    detected_language = getattr(info, "language", None)
    language_probability = getattr(info, "language_probability", None)
    try:
        language_probability = float(language_probability)
        if not math.isfinite(language_probability):
            language_probability = None
    except (TypeError, ValueError):
        language_probability = None
    message = "Lyrics transcribed successfully. Singing may affect recognition and word timing."
    if not words:
        message = "Transcript recognized, but no valid word-level timestamps were provided by the transcription model."
    return {
        "language": detected_language,
        "transcript": transcript,
        "words": words,
        "metadata": {
            "model_size": model_size,
            "duration_seconds": duration,
            "language_probability": language_probability,
            "word_timestamps_requested": True,
            "skipped_word_timestamps": skipped_word_timestamps,
        },
        "message": message,
    }

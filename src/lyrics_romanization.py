"""Romanize lyric text while leaving its language, words, and timing intact."""
from __future__ import annotations

import logging
import unicodedata
from functools import lru_cache
from typing import Any

LOGGER = logging.getLogger(__name__)

# faster-whisper reports ISO-639-1 codes; uroman accepts ISO-639-3 codes.
_UROMAN_LANGUAGE_CODES = {
    "hi": "hin",  # Hindi, usually Devanagari
    "ur": "urd",  # Urdu, usually Perso-Arabic
}


@lru_cache(maxsize=1)
def _romanizer() -> Any:
    import uroman

    return uroman.Uroman()


def _has_non_latin_letters(text: str) -> bool:
    return any(
        char.isalpha() and "LATIN" not in unicodedata.name(char, "")
        for char in text
    )


def _contains_script(text: str, script: str) -> bool:
    return any(script in unicodedata.name(char, "") for char in text if char.isalpha())


def romanize_text(text: str, language: str | None = None) -> str:
    """Return a Latin-script rendering, or the original if conversion is unsupported.

    Romanization preserves the source language; it is not translation. ASCII/Latin
    text is returned untouched. A failed or incomplete conversion also returns the
    original text so unsupported scripts are never replaced with guessed lyrics.
    """
    if not isinstance(text, str) or not text or not _has_non_latin_letters(text):
        return text
    has_arabic = _contains_script(text, "ARABIC")
    has_devanagari = _contains_script(text, "DEVANAGARI")
    if (language or "").lower() in {"ur", "urd"} and has_arabic:
        try:
            # Urdu is normally written without short-vowel marks. This language-
            # specific, context-aware romanizer restores common spoken vowels.
            from urdunlp import transliterate_to_roman

            converted = transliterate_to_roman(text)
            if isinstance(converted, str) and converted.strip() and not _has_non_latin_letters(converted):
                return converted
        except Exception:
            LOGGER.exception("Could not use the Urdu-specific romanizer; trying universal romanization")
    lang = (language or "").lower()
    language_code = _UROMAN_LANGUAGE_CODES.get(lang) if (
        (lang in {"ur", "urd"} and has_arabic)
        or (lang in {"hi", "hin"} and has_devanagari)
    ) else None
    try:
        converted = _romanizer().romanize_string(text, lcode=language_code)
    except Exception:
        LOGGER.exception("Could not romanize lyric text for detected language %r", language)
        return text
    if not isinstance(converted, str) or not converted.strip() or _has_non_latin_letters(converted):
        LOGGER.warning("Romanizer did not fully convert lyric text for detected language %r", language)
        return text
    return converted


def romanize_transcription(result: dict[str, Any]) -> dict[str, Any]:
    """Copy a transcription result with romanized transcript and timed word text.

    Original transcript and per-word forms are retained for debugging. Timing and
    every other word field (including probability) are copied without modification.
    """
    language = result.get("language")
    romanized = dict(result)
    original_transcript = result.get("transcript", "")
    romanized["original_transcript"] = original_transcript
    romanized["transcript"] = romanize_text(original_transcript, language)

    words = []
    for original in result.get("words", []):
        word = dict(original)
        original_word = word.get("word", "")
        converted_word = romanize_text(original_word, language)
        if converted_word != original_word:
            word["original_word"] = original_word
        word["word"] = converted_word
        words.append(word)
    romanized["words"] = words
    return romanized

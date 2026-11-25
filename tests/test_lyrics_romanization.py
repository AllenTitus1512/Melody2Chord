"""Romanization tests use real offline transliteration libraries, not lyric fixtures."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from src.lyrics_romanization import romanize_text, romanize_transcription


class LyricsRomanizationTests(unittest.TestCase):
    def test_english_text_is_preserved(self):
        self.assertEqual(romanize_text("I love you", "en"), "I love you")

    def test_hindi_is_converted_to_latin_script(self):
        result = romanize_text("मैं तुमसे प्यार करता हूँ", "hi")
        self.assertTrue(result)
        self.assertTrue(all(ord(char) < 128 for char in result if char.isalpha()))
        self.assertNotEqual(result, "मैं तुमसे प्यार करता हूँ")

    def test_urdu_uses_language_aware_romanization(self):
        self.assertEqual(romanize_text("تم میرا دل ہو", "ur"), "tum mera dil ho")

    def test_transcription_words_keep_timing_and_probability(self):
        source = {
            "language": "ur",
            "transcript": "تم میرا دل ہو",
            "words": [{"word": "تم", "start": 0.1, "end": 0.4, "probability": 0.9}],
            "metadata": {"language_probability": 0.62},
        }
        result = romanize_transcription(source)
        self.assertEqual(result["transcript"], "tum mera dil ho")
        self.assertEqual(result["original_transcript"], source["transcript"])
        self.assertEqual(result["words"][0]["word"], "tum")
        self.assertEqual(result["words"][0]["original_word"], "تم")
        self.assertEqual(result["words"][0]["start"], 0.1)
        self.assertEqual(result["words"][0]["end"], 0.4)
        self.assertEqual(result["words"][0]["probability"], 0.9)
        self.assertEqual(result["metadata"], source["metadata"])

    def test_unsupported_or_failed_conversion_retains_original(self):
        self.assertEqual(romanize_text("🧿", "xx"), "🧿")
        with patch("uroman.Uroman.romanize_string", side_effect=RuntimeError("test failure")):
            self.assertEqual(romanize_text("नमस्ते", "xx"), "नमस्ते")


if __name__ == "__main__":
    unittest.main()

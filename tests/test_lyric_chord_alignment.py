"""Alignment unit tests use mocked timestamps/labels, not model predictions."""
from __future__ import annotations

import unittest
from pathlib import Path

from src.lyric_chord_alignment import align_lyrics_and_chords, trained_chord_classes

ROOT = Path(__file__).resolve().parents[1]
CLASSES = trained_chord_classes()


class LyricChordAlignmentTests(unittest.TestCase):
    def test_chord_uses_closest_overlapping_word_and_silence_is_not_forced(self):
        words = [
            {"word": "I", "start": 0.8, "end": 1.0},
            {"word": "hold", "start": 1.05, "end": 1.6},
            {"word": "on,", "start": 1.7, "end": 2.0},
        ]
        predictions = [
            {"chord": "C", "start": 0.0, "end": 0.5, "probability": 0.7},
            {"chord": "Em", "start": 1.02, "end": 1.8, "probability": 0.6},
        ]
        result = align_lyrics_and_chords(words, predictions, chord_classes=CLASSES)
        self.assertEqual(result["lines"][0]["words"][1]["word"], "hold")
        self.assertEqual([(event["chord"], event["word_index"]) for event in result["lines"][0]["chords"]], [("Em", 1)])
        self.assertEqual(result["stats"]["unmatched_prediction_count"], 1)
        self.assertTrue(any("did not overlap" in warning for warning in result["warnings"]))

    def test_adjacent_repeated_chord_is_suppressed_but_return_is_kept(self):
        words = [{"word": f"w{i}", "start": i * 0.5, "end": (i + 1) * 0.5} for i in range(6)]
        predictions = [
            {"chord": "C", "start": 0.0, "end": 0.5, "probability": 0.5},
            {"chord": "C", "start": 0.5, "end": 1.0, "probability": 0.4},
            {"chord": "Am", "start": 1.0, "end": 1.5, "probability": 0.6},
            {"chord": "C", "start": 1.5, "end": 2.0, "probability": 0.7},
        ]
        result = align_lyrics_and_chords(words, predictions, chord_classes=CLASSES)
        self.assertEqual([event["chord"] for event in result["lines"][0]["chords"]], ["C", "Am", "C"])
        self.assertEqual(result["stats"]["suppressed_repeated_chord_count"], 1)
        self.assertEqual(len(result["prediction_details"]), 4)
        self.assertEqual(result["lines"][0]["chords"][0]["end"], 1.0)

    def test_punctuation_and_long_words_wrap_without_losing_order(self):
        words = [
            {"word": "Honestly,", "start": 0.0, "end": 0.4},
            {"word": "extraordinarily", "start": 0.4, "end": 1.0},
            {"word": "long", "start": 1.0, "end": 1.2},
            {"word": "words.", "start": 1.2, "end": 1.5},
            {"word": "Next", "start": 1.6, "end": 1.9},
            {"word": "line!", "start": 1.9, "end": 2.2},
        ]
        result = align_lyrics_and_chords(words, [], chord_classes=CLASSES, max_line_chars=18)
        flattened = [word["word"] for line in result["lines"] for word in line["words"]]
        self.assertEqual(flattened, [word["word"] for word in words])
        self.assertEqual(result["lines"][-1]["words"][-1]["word"], "line!")
        self.assertTrue(any(line["words"][-1]["word"] == "words." for line in result["lines"]))

    def test_empty_lyrics_and_empty_chords_are_graceful(self):
        empty = align_lyrics_and_chords([], [], chord_classes=CLASSES)
        self.assertEqual(empty["lines"], [])
        self.assertTrue(any("No valid timed lyric words" in warning for warning in empty["warnings"]))
        lyrics_only = align_lyrics_and_chords(
            [{"word": "Hello!", "start": 0, "end": 0.5}], [], chord_classes=CLASSES)
        self.assertEqual(lyrics_only["lines"][0]["words"][0]["word"], "Hello!")
        self.assertEqual(lyrics_only["lines"][0]["chords"], [])

    def test_invalid_timestamps_unsupported_classes_and_order_are_reported(self):
        words = [
            {"word": "first", "start": 1.0, "end": 1.4},
            {"word": "bad", "start": float("nan"), "end": 1.8},
            {"word": "second", "start": 0.5, "end": 0.9},
        ]
        predictions = [
            {"chord": "C", "start": float("inf"), "end": 1.2, "probability": 0.5},
            {"chord": "F#m", "start": 0.5, "end": 0.9, "probability": 0.5},
            {"chord": "Am", "start": 0.5, "end": 0.9, "probability": 0.5},
        ]
        result = align_lyrics_and_chords(words, predictions, chord_classes=CLASSES)
        self.assertEqual([word["word"] for word in result["lines"][0]["words"]], ["first", "second"])
        self.assertEqual([event["chord"] for line in result["lines"] for event in line["chords"]], ["Am"])
        self.assertTrue(any("invalid text/timestamps" in warning for warning in result["warnings"]))
        self.assertTrue(any("not chronological" in warning for warning in result["warnings"]))
        self.assertTrue(any("trained class list" in warning for warning in result["warnings"]))

    def test_chords_are_sorted_chronologically_and_probability_is_preserved(self):
        words = [{"word": "one", "start": 0, "end": 1}, {"word": "two", "start": 1, "end": 2}]
        predictions = [
            {"chord": "Am", "start": 1, "end": 2, "probability": 0.62},
            {"chord": "C", "start": 0, "end": 1, "probability": 0.88},
        ]
        result = align_lyrics_and_chords(words, predictions, chord_classes=CLASSES)
        self.assertEqual([event["chord"] for event in result["lines"][0]["chords"]], ["C", "Am"])
        self.assertEqual(result["lines"][0]["chords"][0]["probability"], 0.88)

    def test_alignment_endpoint_and_responsive_sheet_markup(self):
        import app as flask_app
        client = flask_app.app.test_client()
        result = client.post("/api/align-lyrics-chords", json={
            "words": [{"word": "Hello!", "start": 0.0, "end": 0.5}],
            "predictions": [{"chord": "C", "start": 0.0, "end": 0.5, "probability": 0.8}],
        })
        self.assertEqual(result.status_code, 200, result.get_json())
        self.assertEqual(result.get_json()["alignment"]["lines"][0]["chords"][0]["chord"], "C")
        rendered = client.get("/")
        self.assertEqual(rendered.status_code, 200)
        self.assertEqual(rendered.get_data(as_text=True).count('id="generate-sheet-button"'), 1)
        html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        css = (ROOT / "static" / "css" / "style.css").read_text(encoding="utf-8")
        js = (ROOT / "static" / "js" / "script.js").read_text(encoding="utf-8")
        self.assertNotIn("Word timestamps", html)
        self.assertNotIn("Prediction Details", html)
        self.assertNotIn("LYRICS EXTRACTION", html)
        self.assertNotIn("Extract Lyrics", html)
        self.assertNotIn("prediction-details", html + css + js)
        self.assertIn('id="lyrics-chord-sheet"', html)
        self.assertIn('id="generate-sheet-button"', html)
        self.assertIn("Generate Chord Sheet", html)
        self.assertEqual(html.count('id="generate-sheet-button"'), 1)
        self.assertIn('id="extract-button"', html)
        self.assertIn("Extract Melody", html)
        self.assertIn('id="detect-key-button"', html)
        self.assertIn("Detect Key", html)
        self.assertIn("@media(max-width:600px){.chord-sheet", css)
        self.assertIn("label.textContent = event.chord", js)
        self.assertIn('fetch("/api/generate-chord-sheet"', js)
        self.assertIn('fetch("/api/extract-melody"', js)
        self.assertIn('fetch("/api/detect-key"', js)
        self.assertIn('setStatus(chordStatus, "Generating chord sheet..."', js)
        self.assertIn('scrollIntoView({ behavior: "smooth"', js)


if __name__ == "__main__":
    unittest.main()

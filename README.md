# Melody2Chord

Melody2Chord is a Flask project for vocal melody upload, pitch extraction, key detection, and future learned guitar-chord suggestions. The web interface remains limited to Steps 1–3; dataset preparation does not change Flask routes or add chord output.

## Run the web application

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Open `http://127.0.0.1:5000`.

## Step 4A — POP909 Dataset Preparation

[POP909](https://github.com/music-x-lab/POP909-Dataset) is an official 909-song pop arrangement dataset. Its numbered song folders provide a MIDI arrangement with named `MELODY`, `BRIDGE`, and `PIANO` tracks, plus time-aligned chord and key annotations. We use its real annotations to create melody-to-harmonizing-chord examples—never synthetic or random data.

Download or clone the official repository, then place the official `POP909` song folders here:

```text
data/raw/POP909/001/
data/raw/POP909/002/
...
```

A cloned repository layout is also accepted:

```text
data/raw/POP909/POP909/001/
```

For each song, `src/prepare_pop909_dataset.py` reads the MIDI `MELODY` instrument using `pretty_midi`, extracts MIDI pitch, scientific note name, standard Hz frequency, note timing, and duration. It aligns notes to each `chord_midi.txt` interval by their actual temporal overlap. It obtains the key for the same interval from `key_audio.txt`, selecting the key annotation with the greatest overlap.

Only direct mappings to the initial supported chord classes (`C`, `Dm`, `Em`, `F`, `G`, `Am`, `A`, `D`, `E`, `Bm`, `G7`, `C7`) are retained. Complex chords, inversions, extensions, `N`, and unsupported roots are skipped and recorded in the statistics file rather than being simplified or invented.

The generated real-data CSV files are:

```text
data/processed/pop909_segments.csv
data/training/melody_chord_dataset.csv
```

They contain song ID, aligned note names, frequencies, overlap durations, key, normalized chord target, and chord start/end times. `song_id` must be retained for later song-level train/test splitting, preventing data leakage across rows of the same song.

Run preparation and read-only validation from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/prepare_pop909_dataset.py
python src/validate_melody_chord_dataset.py
```

Preparation writes `data/processed/pop909_dataset_statistics.json`, including song counts, skipped rows, unsupported labels, class distribution, and segment durations. The validator never modifies the CSV.

Chord annotations are training targets selected by the POP909 annotations. Multiple harmonizations can be musically plausible; a later model will learn dataset patterns, not claim that any melody has one objectively correct chord.

## ML boundary

The Random Forest baseline code from Step 4 remains present but is **not trained in Step 4A**. Model training and Flask chord suggestions are deferred until the POP909-derived data has been reviewed and is sufficient for a song-level evaluation split.

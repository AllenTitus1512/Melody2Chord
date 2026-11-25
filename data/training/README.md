# Real training dataset format

Place a real, human-labeled CSV named `melody_chord_dataset.csv` in this folder. The training script will not generate or infer labels.

Required columns:

```csv
melody_notes,musical_key,chord_label
"G4,A4,B4,D5",G Major,G
```

Optional columns add measured timing and pitch details when available:

```csv
melody_notes,note_frequencies,note_durations,musical_key,chord_label
"A4,C5,E5","440.0,523.25,659.25","0.30,0.45,0.35",C Major,Am
```

- `melody_notes`: comma-separated scientific note names from the melody segment, such as `C4,E4,G4`.
- `note_frequencies`: optional comma-separated positive Hz values; must match the note count when supplied.
- `note_durations`: optional comma-separated positive seconds; must match the note count when supplied.
- `musical_key`: one of the chromatic roots followed by `Major` or `Minor`, for example `F# Minor`.
- `chord_label`: a human-provided suitable/harmonizing chord label. Initially allowed: `C`, `Dm`, `Em`, `F`, `G`, `Am`, `A`, `D`, `E`, `Bm`, `G7`, `C7`.

Each row must represent a real melody-to-harmonizing-chord relationship, sourced and labeled deliberately. Do not insert synthetic, random, or duplicated rows. The trainer reports invalid and duplicate rows and stops when each available class lacks enough real examples for a meaningful stratified evaluation.

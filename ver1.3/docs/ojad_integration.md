# OJAD integration

## Purpose

OJAD is used as a **verified linguistic target source**, not as an audio source.

The system keeps three concepts separate:

- `pitch target`
  - expected H/L pattern and accent-phrase structure
- `reference audio`
  - optional waveform used for playback or alignment support
- `scoring confidence`
  - whether the available evidence is strong enough for feedback

This prevents a natural-sounding TTS waveform from being treated as ground truth.

## Why exact-text overrides

OJAD provides useful sentence-level accent and intonation analysis, but its
automatic text analysis is not guaranteed to be correct for every sentence.
For that reason, the project uses OJAD through a manually reviewed exact-text
table:

```text
configs/verified_accent_targets.json
```

Lookup order:

1. exact-text verified target
2. OpenJTalk accent-phrase fallback
3. heuristic fallback

When an entry uses `pitch_target_source: "ojad_verified"`, prosody scoring may
treat the H/L target as stronger evidence than an automatic OpenJTalk target.

## Import workflow

After checking a sentence in OJAD and verifying the analysis:

```bash
python scripts/import_verified_target.py \
  --text "<exact sentence text>" \
  --kana "<verified katakana reading>" \
  --target-pitch "<one reviewed H/L label per mora>" \
  --source ojad_verified \
  --phrase-lengths "<comma-separated mora counts per accent phrase>" \
  --accent-positions "<comma-separated accent positions>" \
  --note "Checked against OJAD Suzuki-kun on YYYY-MM-DD"
```

Required fields:

- exact text
- kana reading
- one H/L label per mora

Recommended fields:

- phrase lengths
- accent positions
- a provenance note with verification date

## Non-goals

- Do not scrape OJAD audio into this project.
- Do not call TTS-generated speech `ground truth`.
- Do not mark automatically generated, unreviewed targets as
  `ojad_verified`.

## Next useful step

Build a small curated sentence bank with:

- fixed practice text
- OJAD-reviewed pitch target
- native-speaker reference recording when available
- explicit provenance for every target

That bank is more valuable for scoring quality than replacing one TTS voice with
another.

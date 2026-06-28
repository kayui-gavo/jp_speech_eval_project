# Release: Four-dimension Japanese practice package

- date: 2026-06-28
- package: `jp-speech-eval 1.6.0`
- scoring implementation base: `c39ac12`
- release tag: `stable/2026-06-28-four-dimension-practice-package`

## Product Position

This is a Japanese speech practice scoring and evidence-based feedback demo. It
is not an exam-grade or teacher-grade pronunciation assessment system.

The stable learner-facing contract provides four practice dimensions:

- pronunciation clarity
- rhythm / special mora
- fluency
- pitch movement naturalness

## Release Scope

- Packages runtime JSON defaults inside the wheel.
- Declares core and optional Python dependencies in `pyproject.toml`.
- Keeps `user_facing` as the only formal learner UI contract.
- Documents the four dimension keys and confidence fields.
- Adds a Chinese integration handoff guide.
- Adds a reproducible builder for wheel, sdist and a checksum-protected handoff ZIP.
- Includes one fixed-reference sample asset set in the handoff ZIP.

No scoring formula, threshold, aggregate weight, ASR/TTS provider or UI
structure is changed by the packaging work.

## Safety Contract

- Content mismatch and clearly non-Japanese input must not show formal scores.
- A missing `display_score` must never fall back to raw `total_score`.
- A missing user-facing pitch score must never fall back to raw `prosody_score`.
- `tone_score` remains outside the core four dimensions.
- Raw/debug fields are retained for engineering logs only.

## Verification

- Full repository test baseline before packaging: `259 passed, 4 warnings`.
- Package-specific tests cover resource portability and public API metadata.
- The release procedure includes isolated wheel installation and sample API smoke testing.

## Known Limitations

- No large-scale human-listener calibration.
- Strict pitch accent correctness is not established.
- Wrong accent-drop detection remains limited.
- Natural dialogue, dialect, emotion and noisy-channel coverage remains limited.
- Special-mora feedback still depends strongly on boundary evidence.
- Arbitrary-sentence evaluation requires user-confirmed Japanese text.

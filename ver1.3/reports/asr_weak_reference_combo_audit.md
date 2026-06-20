# ASR weak-reference combo audit

- generated_at: 2026-06-20T08:13:10+00:00
- rows: 36
- mode: `asr_confirmed_weak_reference`
- external provider calls: 0
- API keys required: no
- evidence class: offline plan replay; candidate transcripts are fixtures, not empirical provider outputs.
- TTS setting: fixed current demo pseudo-reference; TTS is not the comparison variable.

## Results

| combo | normalized match | Japanese accept | bad reject | kana similarity | hallucination risk | good visible if confirmed | no-score correctness | timestamps useful | feedback location useful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_asr | 0.5 | 0.8333 | 0.6 | 0.9556 | 0.4 | 0.8333 | 1.0 | 0.0 | 0.0 |
| best_candidate_asr | 1.0 | 1.0 | 0.8 | 1.0 | 0.2 | 1.0 | 1.0 | 1.0 | 1.0 |
| oracle_transcript | 1.0 | 1.0 | 0.8 | 1.0 | 0.2 | 1.0 | 1.0 | 0.0 | 0.0 |

## Gaps

- Best-candidate minus current Japanese accept: 0.1667.
- Best-candidate minus current normalized match: 0.5000.
- Oracle minus best Japanese accept: 0.0000.
- Oracle minus best bad-input reject: 0.0000.

## Answers

- The best-candidate slot looks better than current ASR on good-Japanese eligibility and provides timestamp anchors. Statistical significance cannot be claimed: the fixture candidates are deliberately near-oracle and no real provider processed the audio.
- Oracle still leaves a bad-input edge case in this synthetic plan, showing that transcript correctness alone does not replace language/content evidence and user confirmation.
- ASR is a primary ceiling for arbitrary-speech eligibility, kana/mora stability, and feedback localization. It is not the only product ceiling: F0 extraction, timing proxies, learner calibration, and pedagogy remain downstream limitations.
- TTS is not wholly absent from the current implementation: the fixed pseudo-reference cache can indirectly influence alignment/timing proxies. It is held constant here and is not a strict pitch judge in weak native-likeness scoring.

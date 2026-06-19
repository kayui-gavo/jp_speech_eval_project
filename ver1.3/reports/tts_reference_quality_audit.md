# TTS reference quality audit

- generated_at: 2026-06-19T20:38:43+00:00
- plan: `/Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/data/tts_benchmark_plan.csv`
- matrix rows: 50
- local fixture rows measured: 2
- external provider calls: 0
- status: provider matrix and local-cache replay; missing external outputs remain explicitly unavailable.

## Available local fixtures

| case | provider | duration | F0 coverage | timing | pitch range | movement | kana match | reliability |
|---|---|---:|---:|---|---:|---:|---:|---|
| ramen_kudasai | current_pyopenjtalk | 1.088 | 1.0 | approximate | 0.349 | 0.9172 | 1.0 | weak |
| jvs_selected_014 | human_native_reference_oracle | 2.432 | 1.0 | non_fallback | 0.7815 | 0.4599 | 1.0 | reliable |

## Interpretation

- Better TTS may improve reference-audio naturalness, shadowing UX, phrase timing, and how easy the demo is to imitate.
- TTS output always remains synthetic/unverified and weak for scoring provenance, regardless of subjective quality.
- Existing verified JVS human audio is an oracle comparator, not a TTS provider result.
- Better TTS alone cannot create human/native pitch ground truth, calibrate learner scores, or validate wrong accent drops.
- Real A/B requires generated WAV files for the same text/voice/style conditions, followed by the offline metrics in this matrix and human listening review.

# ASR bottleneck audit

- generated_at: 2026-06-19T20:39:55+00:00
- plan: `/Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/data/asr_benchmark_plan.csv`
- replay rows: 72
- external provider calls: 0
- status: offline planning replay, not an empirical provider benchmark.

## Provider-slot summary

| provider | good accept | bad reject | hallucination risk | mean kana similarity | false reject | false accept if blindly confirmed | timestamp cases |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_asr | 0.8333 | 0.6 | 0.4 | 0.9556 | 1 | 2 | 0 |
| oracle_transcript | 1.0 | 0.8 | 0.2 | 1.0 | 0 | 1 | 0 |
| whisper_large_or_faster_whisper | 1.0 | 0.8 | 0.2 | 1.0 | 0 | 1 | 0 |
| whisperx_alignment | 1.0 | 0.8 | 0.2 | 1.0 | 0 | 1 | 12 |
| openai_transcribe_candidate | 1.0 | 0.8 | 0.2 | 1.0 | 0 | 1 | 12 |
| manual_transcript_confirmed | 1.0 | 1.0 | 0.0 | 1.0 | 0 | 0 | 0 |

## Interpretation

- Better ASR can improve candidate transcript quality, kana/mora stability, content-gate decisions, and the amount of manual correction needed.
- Word/segment timestamps may improve coarse feedback location and alignment initialization, especially for pauses and special-mora sentences.
- User confirmation remains the product safety boundary. A Japanese-looking hallucination must not become a formal score merely because an ASR provider emitted it.
- Oracle/manual transcript replay improves noisy or pronunciation-variant content eligibility, but it does not improve F0 extraction or pitch naturalness features.
- This replay cannot rank real providers until identical audio is transcribed offline by each candidate and stored with language/confidence/timestamps.

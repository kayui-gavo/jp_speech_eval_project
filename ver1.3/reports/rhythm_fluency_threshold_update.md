# Rhythm and fluency threshold update

Current recommendation: use JVS native percentiles as a guardrail for threshold tuning, but do not automatically overwrite runtime thresholds from this small snapshot.

## Native guardrails
- speech_rate_mora_per_sec: P05=5.6071, P50=6.3517, P95=7.1805 (n=17)
- pause_ratio: P05=0.1055, P50=0.1766, P95=0.3051 (n=17)
- mora_duration_cv: P05=0.0, P50=0.0, P95=0.0 (n=17)
- rhythm_timing_score: P05=90.2, P50=98.0, P95=100.0 (n=17)
- delivery_fluency_score: P05=66.0, P50=96.0, P95=100.0 (n=17)

## Product rule
- Penalize native-like timing less aggressively.
- Split rhythm timing from delivery fluency so a fast but clear utterance is not forced into the same bucket as a choppy utterance.
- Keep pitch-accent diagnosis out of the display score unless the target is verified and the signal evidence is strong.

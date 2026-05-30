# Special mora calibration report

This report calibrates special-mora feedback with native JVS sanity checks. It is threshold evidence, not a trained model.

## Native coverage and distribution
- long_vowel: count=117, reliable_count=117, mean=0.5749, P05=0.2307, P50=0.51, P95=1.0948, fallback=0.0, uncertain=0.0, sufficient_evidence=True
- sokuon: count=12, reliable_count=12, mean=0.3945, P05=0.2114, P50=0.3438, P95=0.7794, fallback=0.0, uncertain=0.0, sufficient_evidence=False
- moraic_nasal: count=44, reliable_count=44, mean=0.6284, P05=0.2742, P50=0.6324, P95=1.1429, fallback=0.0, uncertain=0.0, sufficient_evidence=True
- yoon: count=66, reliable_count=66, mean=1.4762, P05=0.8804, P50=1.425, P95=2.2489, fallback=0.0, uncertain=0.0, sufficient_evidence=True

## Proposed thresholds
- long_vowel: low=0.2307, high=1.0948 (P05/P95 conservative guardrail)
- sokuon: insufficient reliable alignment evidence; keep default/evidence-gated behavior
- moraic_nasal: low=0.2742, high=1.1429 (P05/P95 conservative guardrail)
- yoon: low=0.8804, high=2.2489 (P05/P95 conservative guardrail)

## Limitations
- If JVS coverage is small for a type, runtime feedback must stay conservative or uncertain.
- Transcript-based filtering can miss lexical details, especially weak long-vowel spellings.
- Equal/fallback alignment is not safe for concrete special-mora correction.
- TTS and Kanade references are pseudo references and are not used as native ground truth.

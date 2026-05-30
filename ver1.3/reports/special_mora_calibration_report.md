# Special mora calibration report

This report calibrates special-mora feedback with native JVS sanity checks. It is threshold evidence, not a trained model.

## Native coverage and distribution
- long_vowel: count=117, reliable_count=0, mean=None, P05=None, P50=None, P95=None, fallback=0.0, uncertain=1.0, sufficient_evidence=False
- sokuon: count=12, reliable_count=0, mean=None, P05=None, P50=None, P95=None, fallback=0.0, uncertain=1.0, sufficient_evidence=False
- moraic_nasal: count=44, reliable_count=0, mean=None, P05=None, P50=None, P95=None, fallback=0.0, uncertain=1.0, sufficient_evidence=False
- yoon: count=66, reliable_count=0, mean=None, P05=None, P50=None, P95=None, fallback=0.0, uncertain=1.0, sufficient_evidence=False

## Proposed thresholds
- long_vowel: insufficient reliable alignment evidence; keep default/evidence-gated behavior
- sokuon: insufficient reliable alignment evidence; keep default/evidence-gated behavior
- moraic_nasal: insufficient reliable alignment evidence; keep default/evidence-gated behavior
- yoon: insufficient reliable alignment evidence; keep default/evidence-gated behavior

## Limitations
- If JVS coverage is small for a type, runtime feedback must stay conservative or uncertain.
- Transcript-based filtering can miss lexical details, especially weak long-vowel spellings.
- Equal/fallback alignment is not safe for concrete special-mora correction.
- TTS and Kanade references are pseudo references and are not used as native ground truth.

# Runtime special mora shadow report

## Threshold status
- long_vowel: status=active, low=0.2307, high=1.0948, sample_count=117, source=JVS
- moraic_nasal: status=active, low=0.2742, high=1.1429, sample_count=44, source=JVS
- sokuon: status=insufficient, low=None, high=None, sample_count=12, source=JVS
- vowel_lengthening_candidate: status=invalid, low=None, high=None, sample_count=None, source=None
- yoon: status=debug_only, low=0.8804, high=2.2489, sample_count=66, source=JVS

## Runtime shadow decisions
- total decisions: 41
- user_feedback_allowed by type: {}
- suppressed by reason: {'no_correction_needed': 23, 'missing_or_invalid_threshold_metadata': 6, 'insufficient_native_evidence': 6, 'debug_only_threshold': 6}
- sokuon/yoon user-facing leakage: 0
- display score safety: pass: unavailable special_mora_score is not emitted as zero
- JVS native false alarm under runtime rows: 0

## Sample audit trend inputs
- JVS: rows=275, by_type={'long_vowel': 130, 'yoon': 74, 'moraic_nasal': 52, 'sokuon': 19}
- JVS: offline sample decisions={'ok': 144, 'uncertain': 114, 'too_short': 9, 'too_long': 8}
- JANON: rows=36, by_type={'sokuon': 7, 'moraic_nasal': 8, 'yoon': 8, 'long_vowel': 13}
- JANON: offline sample decisions={'uncertain': 36}

## Allowed feedback examples
- none

## Suppressed examples
- basic_ramen: long_vowel ー suppressed=no_correction_needed
- basic_ramen: moraic_nasal ン suppressed=no_correction_needed
- basic_sumimasen: moraic_nasal ン suppressed=no_correction_needed
- basic_repeat: long_vowel ー suppressed=no_correction_needed
- basic_repeat: vowel_lengthening_candidate オ suppressed=missing_or_invalid_threshold_metadata
- long_vowel_coffee: long_vowel ー suppressed=no_correction_needed
- long_vowel_coffee: long_vowel ー suppressed=no_correction_needed
- long_vowel_supermarket: long_vowel ー suppressed=no_correction_needed

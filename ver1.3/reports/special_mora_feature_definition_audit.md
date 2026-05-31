# Special mora feature definition audit

This report defines the acoustic proxy behind each special-mora feature before any user-facing threshold is trusted.

## long_vowel
- feature: long_vowel_ratio_to_avg_mora
- definition: duration of the mora mapped to the long vowel mark / vowel-lengthening mora, measured from reliable phone-label-backed mora segment
- denominator: mean duration of mapped mora segments in the utterance; pauses are excluded by phone label parser, special mora are currently included
- interpretation: low means the lengthening mora is short relative to this utterance's mora timing; high means it may be over-held
- threshold status: active
- low/high: 0.2307 / 1.0948
- warnings: None

## sokuon
- feature: closure_ratio_to_neighbor_mora
- definition: duration of the sokuon mora segment mapped from phone labels; if labels lack explicit closure, this is a grouping proxy
- denominator: mean duration of previous and next mapped mora segments
- interpretation: tentative only until enough samples verify whether the label captures closure/geminate timing
- threshold status: insufficient
- low/high: None / None
- warnings: insufficient reliable alignment evidence

## moraic_nasal
- feature: nasal_ratio_to_avg_mora
- definition: duration of the moraic nasal mora segment, often mapped to /N/ or its allophonic label in JVS lab
- denominator: mean duration of mapped mora segments in the utterance; pauses excluded, special mora included
- interpretation: low can indicate a collapsed nasal or mapper under-segmentation; high can indicate over-held nasal
- threshold status: active
- low/high: 0.2742 / 1.1429
- warnings: None

## yoon
- feature: yoon_mora_count_consistency
- definition: whether a yoon surface mora is grouped as one mora with its phone cluster; duration is debug-only
- denominator: not a pure duration threshold; duration ratios are reported only for audit
- interpretation: yoon feedback should prioritize one-mora grouping/glide mapping, not too_short/too_long duration
- threshold status: debug_only
- low/high: 0.8804 / 2.2489
- warnings: yoon_duration_threshold_debug_only_use_mora_count_consistency

## Important notes
- Yoon is not primarily a duration problem; duration ratios are debug-only unless later validated.
- Sokuon requires enough real closure/geminate evidence; current low sample count keeps it insufficient.
- Average mora duration currently includes mapped special mora but excludes silence/pause phones removed by the label parser.

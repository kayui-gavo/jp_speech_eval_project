# Prosody wrong-drop sensitivity diagnostics

- generated_at: 2026-06-19T02:42:16+00:00
- pairs: 12
- scope: diagnostic only; no prosody weights or scoring formulas are changed.

## Summary

| field | n | mean | min | p50 | max |
|---|---:|---:|---:|---:|---:|
| score_delta_native_minus_wrong | 12 | 3.8333 | 0.0 | 4.0 | 11.0 |
| accent_drop_target_count | 12 | 0.8333 | 0.0 | 1.0 | 2.0 |
| changed_drop_overlap_count | 12 | 0.0 | 0.0 | 0.0 | 0.0 |
| transition_delta | 12 | 0.0754 | -0.0185 | 0.0758 | 0.1832 |
| approx_transition_score_loss | 12 | 1.8857 | -0.4613 | 1.8952 | 4.5802 |
| contour_delta | 12 | 0.0291 | 0.0042 | 0.0287 | 0.0699 |
| approx_contour_score_loss | 12 | 1.601 | 0.2311 | 1.5805 | 3.842 |

## Weak-Separation Cases

- wrong_drop_score_not_below_native_count: 3/12

## Interpretation

- The wrong-drop counterfactual usually changes the intended drop mora, but the total score only changes modestly.
- Current scoring has contour, transition, final, and H/L components; accent-drop agreement is logged and used for feedback, but it is not an independent weighted score component.
- Wrong-drop therefore only affects the aggregate indirectly through transition direction and contour similarity.
- In many sentences, changing one accent-drop transition leaves the overall contour correlation high, so the score remains close to native.
- OpenJTalk-derived accent phrase/drop targets are still heuristic; drop-specific feedback should require reliable targets.

## Candidate Fixes For Later

- Add a separate accent-drop subscore after target confidence is reliable.
- Detect phrase-level drop windows instead of only adjacent mora transitions.
- Keep accent-drop feedback unavailable when target confidence is weak.
- Validate with real or waveform-level wrong-accent audio before enabling calibration.

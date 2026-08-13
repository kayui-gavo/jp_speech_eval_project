# C-end v2 real-audio acceptance

## Executive verdict

**BLOCK MERGE**

The candidate is runtime-stable on the available Japanese panel and every one of the 40 runnable cases produced a practice score. The real ASR-backed target-mismatch fallback also completed end to end. However, this run cannot establish negative-control safety because the repository contains no labelled English-content, Mandarin-content, silence, or pure-noise recording. More importantly, a real similar-duration wrong Japanese sentence passed the acoustic content gate as verified target content and never invoked ASR. The score distribution is also strongly ceiling-weighted. These are production acceptance gaps, not reasons to tune thresholds in this run.

The shadow runtime defects found during the audit were repaired without changing any production score, threshold, weight, or user-facing mapping. Shadow quality limitations do not independently block the merge because all shadows remain disabled by default.

## Protocol and assets

- Candidate start: `3006bab0ea9c7c940b18a104a9cc7e4a0d2863d8` on `c-end-product-v2-local-fixes`.
- Baseline: `5528de10e17675e715bd64653f5cd8da6d98d5e6` (`origin/main`) in a separate detached clone. The candidate branch was not reset.
- Same absolute WAV paths, cache prefixes, configuration defaults, Python environment, and manifest were used for both versions.
- Real audio sources: JVS `parallel100`, JANON native and learner fixed-reading, the existing ramen demo recording, and existing Batch 3 edits of real JVS speech.
- The pitch-shift file is the one explicitly allowed audit-only +3-semitone invariance transform. It was not used as training data.
- Inventory and reproducibility files are under `ver1.3/outputs/c_end_v2_acceptance/`.

Acceptance matrix (40 runnable samples):

| Category | n | Coverage |
|---|---:|---|
| Native Japanese | 17 | JVS cross-speaker/self and JANON `jpf1` |
| Learner Japanese | 14 | JANON Mandarin-L1 and English-L1 speakers reading Japanese |
| Real target mismatch | 2 | JVS sentence B assigned target/cache A |
| Direct broad mode | 2 | Same JVS recordings with supplied Japanese transcript |
| Partial/continuity edits | 2 | Existing local-delete-like and long-pause edits |
| Moderately poor recording | 1 | Existing 15 dB mild-noise JVS edit |
| Weak-reference paths | 2 | Existing ramen recording, fixed and confirmed-weak flows |
| Labelled non-Japanese / silence / pure noise | 0 | **Coverage gap** |

JANON L1-English and L1-Mandarin rows are learner Japanese recordings, not non-Japanese-content controls. No unknown recording was relabelled to fill a missing category.

## Product scoring

- Reasonable Japanese score availability was `40/40` in the candidate. Native plus learner fixed-reading alone was `31/31`.
- Baseline availability was also `40/40`; therefore old no-score -> new score was `0`. This panel demonstrates retained availability, not an availability improvement over the selected `origin/main` baseline.
- Candidate and baseline display scores were identical on every case. Pronunciation, prosody, fluency, tone, raw total, and alignment mode were also unchanged on all 40 cases.
- One intended UX state changed: `demo_ramen_confirmed_weak` changed from `pass` to `practice_suggestion` while keeping score 72.
- The candidate exposes pronunciation/rhythm/fluency values on all 40 cases. Strict pitch-accent was unavailable on all 40 because none of the active targets had eligible verified lexical pitch evidence. This run therefore does not validate the verified-target pitch display path.
- No evaluator crash occurred.

This preserves the rule that shadow diagnostics must not alter production scores. It also shows that the branch's principal observable changes on this panel are claim/availability semantics rather than score remapping.

## Negative controls

No labelled real English-content, Mandarin-content, nonsense, silence, or pure-noise WAV was present in the scanned repository assets. Consequently:

- false acceptance of non-Japanese cannot be estimated;
- silence/noise retry behavior cannot be accepted;
- old no-score -> new score regression counts for negative controls are unavailable.

The existing mild-noise sample still contains intelligible Japanese and correctly received score 89; it is a channel robustness case, not a negative control. Merge should remain blocked until the missing real controls are supplied and run without changing thresholds based on this panel.

## Target mismatch fallback

One forced large mismatch completed the requested full path:

- WAV: JVS `jvs002/VOICEACTRESS100_002`.
- Assigned target: `ラーメンをください`.
- Fixed content match: `fail`, acoustic DTW cost 3.6581, duration ratio 7.0083.
- ASR: local `faster-whisper-small` (pre-existing checkpoint).
- Transcript: `ニューイングランド風は牛乳をベースとした白いクリームスープであり、ボストンクラムチャウダーとも呼ばれる。`
- Transcript sanity: `ok`, Japanese ratio 1.0.
- Effective mode: `reference_mismatch_general_japanese`.
- Candidate content semantics: `content_verified=false`, `japanese_content_plausible=true`.
- Score: 84. The same WAV run directly in broad mode also scored 84.
- Pronunciation/rhythm/fluency: 88/71/71; strict pitch and target-local detail unavailable.
- User message states that the content differs and that only broad Japanese speaking style was evaluated.

The fallback is therefore real and deterministic, not a mocked `content_match` path. The baseline also reached the broad path and scored 84, but incorrectly marked `content_verified=true`; the candidate fixes that claim.

A second real mismatch exposed a production limitation:

- JVS sentence 002 was assigned JVS sentence 001 with similar duration.
- Acoustic gate returned `pass` with score 0.7506, DTW cost 3.7990, and duration ratio 0.9928.
- Policy skipped ASR, marked `content_verified=true`, remained in `reference_based`, and scored 89.

Thus the fallback works when the current gate produces `fail`, but similar-duration wrong content can still bypass it. This is the main production blocker found by the run.

## Score distribution

Candidate display score distribution over all 40 Japanese/runnable cases:

| group | n | mean | std | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 40 | 89.825 | 8.087 | 72 | 77.8 | 84.0 | 90 | 98 | 100 | 100 |
| native | 17 | 93.235 | 7.321 | 75 | 84.6 | 88.0 | 95 | 100 | 100 | 100 |
| learner | 14 | 89.071 | 8.233 | 75 | 77.8 | 82.5 | 89.5 | 97 | 98.7 | 100 |
| broad | 2 | 81.0 | 3.0 | 78 | 78.6 | 79.5 | 81 | 82.5 | 83.4 | 84 |
| general fallback | 2 | 86.5 | 2.5 | 84 | 84.5 | 85.25 | 86.5 | 87.75 | 88.5 | 89 |
| weak reference | 2 | 81.0 | 9.0 | 72 | 73.8 | 76.5 | 81 | 85.5 | 88.2 | 90 |

`21/40` scores were at least 90 and `7/40` were exactly 100. Native is higher than learner at group level, but the ceiling is substantial and several learner samples reach 97-100. There are no listener labels in this panel, so this is a calibration risk rather than proof that any particular score is wrong. No score mapping was changed.

Mode comparisons:

- The same correctly matched JVS recording scored 90 fixed-reference versus 78 broad, a 12-point mode effect attributable to using target/reference evidence.
- The forced ramen mismatch and direct broad evaluation of the same recording both scored 84, so fallback itself introduced no unexplained jump.
- The same ramen recording scored 90 in fixed mode and 72 in confirmed-weak mode; the 18-point difference follows the weak-reference evidence downgrade and should remain visible in calibration review.

## Fixed vs broad behavior

- Seven fixed cases used `cached_dtw_fallback_equal`; all retained scores (85-90), medium-or-lower confidence, no strict pitch, and no target-local detail.
- The low-F0 demo case had F0 coverage 0.3333 yet retained overall 90 and pronunciation/rhythm/fluency 80/100/100. Pitch was unavailable. This confirms that low F0 does not erase the other dimensions or overall result.
- The existing 15 dB mild-noise recording scored 89 and retained detail.
- The local-delete-like and long-pause samples scored 90 and 85. Both remained broad/fallback-alignment cases, so no unsupported exact local diagnosis was exposed.
- The content gate did not identify the local-delete-like edit as partial content. This is consistent with the known 1-best/acoustic content limitation and should not be interpreted as successful deletion detection.

## SSL pronunciation shadow

The specified checkpoint exists and ran: `microsoft/wavlm-large`, revision `c1423ed...`, 24 transformer layers, hidden size 1024, layer 12 selected, 16 kHz input, CPU execution.

Initial real execution failed after 8.1 s with:

`RuntimeError: Failed to load SSL checkpoint microsoft/wavlm-large: expected str, bytes or os.PathLike object, not NoneType`

Cause: the official WavLM checkpoint provides a feature extractor but not the tokenizer assets expected by `AutoProcessor` in the installed Transformers version. The shadow now uses `AutoFeatureExtractor`, forwards its attention mask/model inputs, and caches the lazy extractor for warm calls. The broad fallback now preserves the fixed cache prefix so mismatch SSL can still compare against the assigned reference. Errors remain isolated from the product score.

Real distances (cosine-frame DTW, path-length normalized; no /100 mapping):

| sample | relation | SSL distance | MFCC/content DTW |
|---|---|---:|---:|
| JVS native self | same speaker/reference | 0.078961 | 0.0832 |
| JANON learner | learner vs native, same target | 0.245396 | 2.9380 |
| JVS native cross-speaker | native vs native, same target | 0.268505 | 3.0143 |
| JVS mismatch vs ramen | different target | 0.569631 | 3.6581 |

Self is closest and mismatch is farthest. The native cross-speaker distance was slightly worse than the learner distance, so the requested full qualitative ordering was not achieved. Cross-target/corpus distances are not calibrated and speaker dependence remains unresolved. The shadow is executable but not ready for a score mapping.

## Special mora v2 shadow

Real target-local evidence was exercised for all three requested types:

- Sokuon: JANON `ばっちり` and `うっとうしい`.
- Long vowel: JANON `うっとうしい` and JVS long-vowel targets.
- Moraic nasal: JANON `酸味`.

No negative duration, out-of-bounds ROI, NaN, or constant all-zero payload occurred. The first run did expose false precision under equal-boundary fallback: ten equal-width long-vowel ROIs were labelled medium-confidence. The shadow now records ROI validity and alignment mode, returns low confidence and `unavailable_alignment_or_roi_unreliable` under `fallback_equal`, and never changes production output.

Remaining scientific limitations:

- Both real sokuon ROIs had `low_energy_fraction=0.0` even though energy-to-neighbor ratios were about 0.052. The relative low-energy fraction detector is not functioning as useful closure evidence on these samples; the neighbor comparison is carrying the signal.
- The long-vowel example had duration 0.18 s and neighbor ratio 0.529, but voicing autocorrelation was -0.660, so continuity evidence is not yet robust.
- The moraic nasal example had duration 0.35 s but voicing autocorrelation -0.135. Current spectral/context evidence is insufficient for a correctness claim.
- `local_ssl_reference_distance` and boundary displacement remain unavailable.

The v2 output remains `evidence_only`, `user_facing=false`, and must not be described as phone correctness.

## Phrase intonation

The real JVS native contour produced 80.4% F0 coverage, 4.981 semitones mean transition movement, and candidate score 72.154. After the allowed offline +3-semitone shift, coverage was 84.8%, movement 4.723, and score 74.215 (delta +2.061). Speaker-median semitone normalization therefore showed reasonable global-pitch invariance on this one case.

This is only a one-file invariance check. The movement statistic remains sensitive to F0 extraction and missing-mora patterns, and the candidate score is not used by production.

## Accent nucleus

The first implementation used `reference_source` as a fallback target provenance and therefore treated JVS/JANON native reference audio as a strong lexical nucleus target. That is scientifically incorrect: native reference identity does not verify OpenJTalk's nucleus label. Provenance now comes from the lexical pitch target (`details.pitch_target_source` or prosody HL target), and only `human_checked`, `ojad_checked`, `ojad_reviewed`, or `manual_verified` are strong.

All real acceptance targets resolved to `openjtalk_accent_phrase_chain`, `heuristic`, or otherwise weak provenance. They now report `weak_target=true`. There was no eligible human/OJAD-reviewed lexical nucleus target in this run, so no meaningful correct/incorrect accuracy is claimed. The +3-semitone case also moved the predicted candidate from mora 33 to 37, reinforcing that this output must remain candidate/low-confidence rather than correctness.

## Latency

| path | cold | warm | notes |
|---|---:|---:|---|
| Production fixed sample | 2.095 s | matrix median 0.353 s; p90 0.612 s | CPU, excludes SSL |
| Real ASR mismatch fallback | — | 4.632 s | includes `faster-whisper-small` |
| WavLM shadow, same JVS sample | 8.644 s wall / 7.187 s SSL | 1.423 s wall / 1.322 s SSL | checkpoint already cached on disk for cold measurement |
| Three local shadows | — | about 0.16 s wall on cross-speaker JVS case | product evaluation dominates wall time |

Disabled SSL imports neither Torch nor Transformers and performs no model work. Shadow exceptions are caught. The configured timeout is currently checked after inference rather than being a hard preemptive deadline; because the shadow is default-off this does not alter production latency, but a future opt-in background execution design is required before enabling it operationally.

## Runtime bugs found

1. WavLM `AutoProcessor` incompatibility caused every real SSL run to fail. Fixed with `AutoFeatureExtractor` and complete model input forwarding.
2. A fresh WavLM extractor/model was constructed per request, preventing warm reuse. Fixed with an audit-only lazy extractor cache.
3. Broad mismatch fallback discarded the fixed cache prefix, so SSL returned `fixed-reference audio is unavailable`. Fixed by preserving the cache prefix in compact fallback debug metadata and consuming it only in the shadow.
4. Equal-boundary fallback special-mora ROIs claimed medium confidence. Fixed by separating ROI/alignment reliability from evidence and suppressing a decision under fallback.
5. Accent nucleus conflated reference-audio identity with verified lexical target provenance. Fixed with an explicit lexical-source allowlist.

All fixes are shadow/debug-only except preservation of one debug cache-prefix field. Production scores and raw scoring formulas are unchanged.

Full regression command from `ver1.3/`:

`python -m pytest -q`

Result: **144 passed, 1 deprecation warning in 1.71 s**. The warning is the pre-existing Python `cgi` deprecation in `scripts/debug_ui.py`. Running from the repository root without adding `ver1.3` to the module path causes three `scripts` import collection errors; the package's established test root is `ver1.3/`.

## Required fixes before merge

1. Add labelled real English-content, Mandarin-content, silence, pure-noise, nonsense, and unusable-recording controls; rerun the exact candidate and baseline without threshold changes.
2. Resolve or explicitly contain the similar-duration wrong-Japanese acoustic false accept. At minimum, the acceptance set must demonstrate that this case reaches transcript sanity/general fallback rather than verified fixed content.
3. Decide a calibration protocol with listener-labelled learner recordings before accepting the current ceiling-heavy distribution. Do not tune against this acceptance set.
4. Add at least one verified lexical pitch/nucleus target if the verified-target accent path is part of the merge claim.

## Research/calibration next steps

- Expand paired native/learner same-sentence SSL comparisons before selecting a layer or distance mapping; quantify speaker identity effects.
- Repair sokuon closure evidence and validate long-vowel/nasal continuity on manually inspected ROIs.
- Validate phrase invariance over several shifts, speakers, and extraction-quality levels.
- Treat accent nucleus as unavailable for OpenJTalk-only targets and obtain human/OJAD-reviewed target labels.
- Keep WavLM and all new diagnostics off the production score path.

## Merge recommendation

**BLOCK MERGE** until real negative controls are present and the demonstrated similar-duration wrong-sentence false accept is contained. The shadow runtime repairs themselves are safe to retain: they are default-off, exception-isolated, do not affect user scores, and have dedicated tests.

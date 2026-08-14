# Japanese phone-GOP manual validation protocol v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Status: controlled human recording / inspection protocol; research-only

## 0. Purpose

This protocol is for the first serious manual validation of the Japanese phone-level pronunciation evidence stack. It is **not** a human calibration study for a final `/100` product score.

The questions are narrower and engineering-oriented:

1. Does the phone-CTC/GOP system react in the correct direction when one intended Japanese phone is deliberately damaged?
2. Does it localize the damage to the intended phone/mora instead of lowering the whole utterance indiscriminately?
3. For categorical substitutions, does the strongest competitor phone look plausible?
4. Are repeated normal takes reasonably stable within the same speaker?
5. Are speed, pause and pitch manipulations mostly reflected in the intended consumer dimensions instead of leaking into clarity?
6. Are Japanese special morae handled by a combination of phone evidence and duration/timing evidence rather than GOP alone?
7. Does the system tolerate legitimate Japanese phonetic variation such as high-vowel devoicing?

The controlled-error stage is useful because classic GOP work has successfully used realistic artificial errors to tune and evaluate phone-level detection. However, a deliberately produced error is **not** equivalent to a naturally occurring learner error. Passing this protocol does not prove learner-error validity. Real L2 learner recordings remain a later required stage.

## 1. Evidence basis for the protocol

### 1.1 Phone errors should be tested as substitution, deletion and insertion

Kawai & Hirose's Japanese/English CALL work explicitly treats nonnative phone errors as insertion, deletion and substitution phenomena. Modern CTC-GOP work makes the same distinction and is motivated partly by the inability of conventional pre-segmented GOP to represent deletion and insertion cleanly.

Therefore the manual battery must contain all three categories, not only substitutions.

### 1.2 Artificial but realistic errors are useful for an early controlled test

Kanters, Cucchiarini & Strik (SLaTE 2009) constructed artificial pronunciation errors from an inventory of frequent learner errors and found that evaluation on those artificial errors approximated performance on real learner data reasonably well. This supports a paired same-speaker stress test as an engineering stage.

But the protocol keeps `intended_error` and `realized_error` separate. The label is not considered true merely because the speaker was instructed to make an error.

### 1.3 Japanese special morae require timing evidence

Kawai & Hirose (Eurospeech 1997; STiLL 1998) show that long vowels, moraic nasals and mora obstruents are strongly duration-dependent, and their CALL system normalizes against speaking rate. Amano & Hirata (Interspeech 2008) and Hirata & Amano (Interspeech 2010) further support closure duration relative to a larger timing unit as a stable cue for singleton/geminate contrast across speaking rates.

Therefore:

- `cl` GOP is not sufficient for 促音;
- repeated-vowel phone evidence is not sufficient for 長音;
- `N` GOP is not sufficient for 撥音;
- relative duration/timing must remain part of the rhythm/special-mora pipeline.

### 1.4 Chinese-speaking learner priority errors

The I-JAS-based study of Chinese-speaking learners reports two notable sokuon errors: dropping the sokuon and replacing it with a long vowel or moraic nasal. Examples reported include `ピクニック → ピクニク`, `バスケット → バスケート`, and `サンドイッチ → サンドインチ`.

Recent work on Chinese-speaking learners' palatalized morae reports five misuse patterns: palatalizing two morae into one, lengthening a short palatalized mora, shortening a long palatalized mora, confusing `ゅ/ょ`, and substituting an L1-like vowel. These are included as a priority module, but they must not be treated as universal errors for every Chinese speaker.

### 1.5 GOP output should not be reduced to one posterior number

The 2025 logit-GOP study reports that logit-based variants can outperform probability-based GOP and that maximum-logit GOP had the strongest alignment with human perception among the tested variants. The current shadow extractor therefore logs posterior and logit evidence, competitor identity and entropy. This protocol evaluates all of them before any product mapping is selected.

## 2. Recording design

### 2.1 Two recording levels

**Level A — phone diagnostic**

Use short words or a fixed carrier sentence. This is the main GOP localization test.

Preferred carrier:

`それは、____です。`

A small natural pause after `は` is allowed. A very similar carrier construction has been used in Japanese singleton/geminate perception experiments. The carrier makes speaking context more stable while retaining a clearly identifiable target word.

**Level B — product transfer**

Use short natural sentences such as `ピクニックに行きます。` and `サンドイッチを食べます。` These check whether a phone effect survives in realistic C-end input.

### 2.2 Recording conditions

Keep the same speaker, microphone, room and approximate mouth-to-mic distance within one session. Record ordinary conversational loudness. Avoid clipping. Do not whisper unless a specific control item asks for unusual phonation.

For each core target record:

- `N1`: normal take;
- `N2`: second normal take, spoken naturally again rather than mechanically copied;
- `E`: intentional error take.

The two normal takes provide a within-speaker noise floor. An error is interesting only if its effect is larger than ordinary N1/N2 variation.

### 2.3 Do not use the intended error as ground truth

Every error take has two separate labels:

- `intended_error`: what the speaker tried to do;
- `realized_error`: what a listener actually heard.

If the speaker tried to replace /f/ with /h/ but the result still sounded like an acceptable Japanese /f/, the clip is **not** a successful /f/→/h/ error example.

## 3. Quick pilot: what to say and how to say it

The CSV manifest in `data/phone_gop_manual_validation_manifest_v1.csv` is the source of truth for clip IDs. The following instructions explain the most important manipulations.

### 3.1 促音 / geminate

Target: `それは、かっこです。`

Normal: pronounce `かっこ` naturally.

Intentional deletion: say it as `かこ`, removing the closure represented by `っ`. Do **not** merely speak faster; specifically remove the extra consonant timing.

Expected evidence: the `cl`/geminate region should lose support or appear as a deletion. Neighboring /k/ and vowels should not all collapse equally.

Additional learner-realistic items:

- `ピクニックに行きます。` → deliberately say `ピクニクに行きます。`
- `バスケットをします。` → deliberately say `バスケートをします。`
- `サンドイッチを食べます。` → deliberately say `サンドインチを食べます。`

These three are specifically motivated by reported Chinese-learner error patterns.

### 3.2 長音 / long vowel

Target: `それは、おばあさんです。`

Intentional shortening: say `おばさん`, removing the second /a/ mora. Keep the surrounding consonants normal.

Target: `それは、しゅじんです。`

Intentional overlengthening: say `しゅうじん` so that the `しゅ` vowel becomes a clear long vowel. This is a strong category-changing control, not a subtle learner simulation.

For special-mora scoring, the desired result is **not** “GOP alone detects everything.” The phone evidence and relative-duration evidence should agree when possible.

### 3.3 撥音 /N/

Target: `それは、みんなです。`

Intentional deletion: say `みな`, removing the moraic nasal interval rather than simply shortening the whole word.

Also record the following contexts normally and with deliberate /N/ deletion:

- `それは、さんぽです。` → `さぽ`
- `それは、りんごです。` → `りご`
- `それは、ほんだなです。` → `ほだな`

These contexts are included because Japanese /N/ has context-dependent phonetic realizations. The phone model should not require one single surface nasal realization to score every context correctly.

### 3.4 拗音 / palatalized mora

Target: `それは、きゃくです。`

Intentional split: say `きやく`, clearly making `き` and `や` separate timing units rather than one `きゃ` mora.

Target: `それは、びょういんです。`

Intentional split: say `びよういん`, producing the full `び・よ・う...` sequence instead of `びょ・う...`.

These are primarily insertion/timing tests, not merely a single-phone substitution test.

### 3.5 Vowel substitution

Target: `それは、ゆめです。`

Intentional substitution: say `よめ`. Preserve the initial /y/ and change only the vowel nucleus from /u/ to /o/ as cleanly as possible.

Expected evidence: the intended vowel phone should drop, and /o/ should become a plausible competitor. Other phones should remain comparatively stable.

### 3.6 /f/ versus /h/

Target: `それは、ふねです。`

Normal: Japanese `ふ`, with the usual bilabial fricative-like articulation.

Intentional /h/-like error: pronounce the first consonant with the same kind of glottal /h/ quality used in `は`, avoiding the lip-frication quality of Japanese `ふ`, while keeping the following vowel and `ね` unchanged.

This is an articulatory distortion/substitution test. If the speaker cannot reliably produce the contrast, mark the take `realized_error=uncertain` rather than forcing it into the benchmark.

### 3.7 /ts/ versus /s/

Target: `それは、つきです。`

Intentional substitution: say `すき` instead. This gives a strong /ts/→/s/ categorical control.

Expected evidence: the /ts/ target region should show a strong competitor close to /s/.

### 3.8 /sh/ versus /s/

Target: `それは、すしです。`

Intentional substitution: keep the written target and vowels, but pronounce the second consonant as an /s/-like sound rather than normal Japanese `し` /sh/-like articulation. In practical terms, keep the tongue configuration closer to `す` for the consonant while still producing the following /i/.

Do not use a take if you cannot hear a clear difference afterward.

### 3.9 Voicing contrasts

Target: `それは、バスです。`

Intentional substitution: say `パス`, changing /b/ to /p/ while preserving the rest.

Target: `それは、かぎです。`

Intentional substitution: say `かき`, changing /g/ to /k/ while preserving the rest.

These are useful because a categorical competitor exists in the Japanese phone inventory.

### 3.10 /r/ distortion — optional, not a primary gate

Target: `それは、からだです。`

Optional intentional distortion: replace the Japanese tap-like /r/ with a sustained English-like [l] or [ɹ] if you can produce it reliably.

The current Japanese phone inventory does not contain a dedicated English /l/ or /ɹ/ class, so competitor identity is not expected to be exact. This item tests whether the target /r/ evidence weakens without catastrophic neighbor damage. It is **not** a required pass criterion.

## 4. Dimension-orthogonality controls

Use the fixed sentence:

`ラーメンをください。`

Record the following while trying to preserve the consonants and vowels:

1. `normal`: ordinary natural reading.
2. `fast`: about 15–25% faster overall; do not intentionally delete phones.
3. `slow`: about 15–25% slower overall; do not exaggerate only the special morae.
4. `pause`: keep local speaking rate normal but insert about 0.6–0.8 s silence between `ラーメンを` and `ください`.
5. `flat_pitch`: keep segmental articulation normal while making the F0 movement intentionally flatter/monotone.
6. `exaggerated_pitch`: preserve segmental articulation and timing as much as possible while exaggerating rises/falls.

Expected qualitative behavior:

- fast/slow should affect rhythm and possibly fluency more than clarity;
- inserted pause should mainly lower fluency;
- flat/exaggerated pitch should mainly affect intonation;
- phone-GOP/clarity should not collapse merely because F0 style changed.

This is a key test of whether the four consumer dimensions have become genuinely less entangled.

## 5. Legitimate-variation negative controls

### 5.1 High-vowel devoicing

Record `すきです。` naturally. Do **not** force every /u/ or /i/ to remain strongly voiced. Japanese high-vowel devoicing is a normal phonetic process in appropriate contexts.

Then record a second take with deliberately careful/full vowel voicing.

Both should remain broadly acceptable. A phone model that gives a severe pronunciation error simply because a native-like high vowel is devoiced is not suitable for direct product feedback without additional normalization.

### 5.2 Normal-repeat stability

For every core target, N1 and N2 are both correct controls. The system should not invent a strong phone error merely because pitch, exact duration or microphone phase changed slightly between two ordinary takes.

## 6. Human inspection procedure

### Pass 1 — blind listening

Rename or randomize clips so the listener does not know whether a take is normal or intentionally wrong.

For each clip record:

- `analyzable`: yes/no;
- `heard_target`: what word/sequence was actually heard;
- `content_recovered`: yes/no;
- `audible_error`: none / substitution / deletion / insertion / distortion / timing / multiple;
- `error_location`: mora/phone if identifiable;
- `severity`: 0–3;
- `listener_confidence`: low/medium/high.

Severity scale:

- 0 = no clearly audible error / acceptable Japanese realization;
- 1 = slight deviation, target remains unambiguous;
- 2 = clear error, but intended target can still be recovered;
- 3 = categorical change or strong distortion; another word/sequence is heard or target recovery is difficult.

`analyzable=no` is **null**, not severity 0.

### Pass 2 — reveal the intended manipulation

After blind labeling, reveal the recording instruction and record:

- `realized_as_intended`: yes/no/uncertain;
- `realized_error_type`;
- `realized_target_phone_or_mora`;
- `notes`.

Only clips with `realized_as_intended=yes` should be used as strong controlled-error positives.

If only one person is available to listen, do Pass 1 after a delay and with randomized filenames. A second Japanese native/near-native listener is preferable later, but is not required for this engineering smoke test.

## 7. What to inspect in GOP output

For every N1/N2/E triplet, inspect at least:

1. `mean_logit_margin` of the intended phone;
2. `max_logit_margin`;
3. posterior GOP margin;
4. target posterior/log probability;
5. entropy;
6. best competitor phone;
7. CTC support-frame count/duration;
8. target phone rank among all phones in the word;
9. adjacent-phone score change;
10. utterance-level summary only after phone-level inspection.

### Qualitative pass expectations

Do **not** freeze numeric thresholds before observing the pilot distribution. The first pass criteria are ordering/localization criteria:

- `clean N1 ≈ clean N2` more closely than either is to a successful strong error;
- target-phone evidence should worsen from clean to error;
- for a categorical substitution, intended target phone should usually become one of the weakest phones in the target word;
- best competitor should be the intended substitute or a phonetically plausible near neighbor when the model inventory supports it;
- neighboring phones should not all drop by the same amount;
- digital gain or mild channel changes should produce less target-phone damage than a successful categorical mispronunciation;
- a pause-only manipulation should not look like a segmental pronunciation collapse.

After the pilot, empirical thresholds can be defined from within-speaker clean-repeat variance and error deltas instead of being invented beforehand.

## 8. Special-mora interpretation rules

### 促音

Require both:

- phone/CTC evidence around `cl` or the consonant transition;
- duration/relative timing evidence.

A short `cl` GOP alone should not prove a geminate error if timing is otherwise native-like, and a correct-looking phone class should not erase an obvious duration failure.

### 長音

Treat the extra mora primarily as timing/sequence evidence. A long vowel is not spectrally a completely different vowel phone. Compare repeated-vowel support and duration relative to neighboring units.

### 撥音

Keep /N/ context-sensitive. Do not demand one acoustic nasal place across `さんぽ`, `りんご`, `ほんだな`, etc. The canonical phonological /N/ is one category but has surface allophony.

## 9. Controlled-error data are not the final learner validation

Passing this protocol supports:

- implementation correctness;
- sensitivity to targeted phone manipulations;
- localization;
- relative robustness;
- dimension disentanglement.

It does **not** prove:

- human-level pronunciation assessment;
- general L2 Japanese mispronunciation detection;
- validity for spontaneous speech;
- a calibrated `0–100` clarity scale;
- fairness across L1, gender, age, channel or speaking style.

The next evidence stage must include naturally occurring learner errors. In particular, a system can look excellent on native or deliberately manipulated speech and still fail on real L2 distortions. This is why the current phone-GOP path remains shadow-only.

## 10. Recommended progression

### Stage A — one-speaker controlled battery

Use the manifest shipped with this protocol. Goal: catch obvious model/implementation failures quickly.

### Stage B — 3–5 speakers

Repeat a reduced 10-target battery across multiple speakers. Include at least one native Japanese speaker if feasible. Goal: check speaker dependency and false alarms.

### Stage C — real learners

Use naturally produced learner speech and manually mark actual phone errors. Do not instruct learners to make errors. Goal: criterion validity of error detection.

### Stage D — product clarity mapping

Only after Stage C should phone evidence be fused with WavLM and ASR intelligibility evidence into a stable consumer clarity score. Mapping and weighting remain product-calibration questions, not GOP theory itself.

## 11. References / rationale anchors

- Witt & Young (1998), phone-level GOP and phone-specific thresholds for CALL.
- Kawai & Hirose (1997, 1998), Japanese long vowel / mora nasal / mora obstruent training using duration and minimal pairs.
- Kawai & Hirose (1998), insertion/deletion/substitution framework for L2 phone errors.
- Amano & Hirata (2008); Hirata & Amano (2010), rate-normalized singleton/geminate timing.
- Kanters, Cucchiarini & Strik (2009), controlled artificial errors as an engineering evaluation method for GOP.
- Cao et al. (Interspeech 2024), CTC phone-level assessment including deletion/insertion and alignment-independent GOP.
- Parikh et al. (Interspeech 2025), logit-based GOP and uncertainty/phone-specific evidence.
- Parikh et al. (Interspeech 2025), phonological-knowledge-constrained alignment-free GOP.
- Hirata (2019), I-JAS Chinese-learner sokuon error patterns.
- Liu (2024), Chinese-learner palatalized-mora misuse patterns.
- Shi et al. (Interspeech 2023), importance of nonnative data and alignment behavior in pronunciation assessment.

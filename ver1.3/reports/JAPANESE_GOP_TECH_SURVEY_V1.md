# Japanese GOP / phone-level pronunciation assessment survey v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Status: literature/technology survey; **no production score promotion in this report**

## Executive verdict

GOP (Goodness of Pronunciation) should be treated as a core missing technology for the C-end **明瞭さ / clarity** dimension. The current ASR-kana/acoustic fallback is useful for continuous product UX, but it is not a substitute for phone-level pronunciation evidence.

The best near-term architecture is not “classic Kaldi GOP only”. The literature has moved from HMM/forced-alignment GOP toward CTC-based and segmentation-free GOP. For this project, the practical path is:

1. canonical Japanese phones from pyopenjtalk;
2. a Japanese phone-CTC acoustic model;
3. several phone-level evidence definitions in parallel (posterior GOP, logit margin/max-logit GOP, self-aligned CTC GOP);
4. benchmark them on native, learner, wrong-target, channel-corrupted, speed/pause and special-mora samples;
5. only after evidence, choose the production clarity backbone;
6. keep WavLM reference comparison and ASR kana agreement as complementary evidence, not replacements for phone evidence.

Recommended first runtime candidate: **`prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`** because it is a 94.4M-parameter Japanese phoneme CTC model and therefore much more deployable than WavLM-large or the 315.6M dual-CTC model. Recommended second benchmark candidate: **`sakasegawa/japanese-wav2vec2-large-hiragana-ctc`**, because it exposes a dedicated intermediate phoneme CTC head and a phone inventory designed around pyopenjtalk-style Japanese phones.

`prj-beatrice/...-v5` exists and is 94.4M, but currently has no model card documenting the final training/evaluation state. It should be benchmarked only after v4, not assumed superior merely because the version number is newer.

## 1. What GOP actually measures

Classic GOP is a phone-level confidence/competition measure. The canonical phone is known from the target transcript. Given the acoustic segment corresponding to that phone, the acoustic model estimates how strongly the segment supports the canonical phone relative to competing phones.

The original Witt/Young line of work used likelihood/posterior-derived phone confidence and phone-specific thresholds, comparing automatic phone scores against human judgments. Kaldi's official GOP recipe later operationalized a neural version in which the log posterior score for the canonical phone is compared with the best competing phone. Kaldi also exports a richer phone-level feature vector containing log phone posteriors and pairwise posterior ratios.

Important consequence for this project:

- MFCC-DTW asks whether two acoustic trajectories are similar;
- WavLM-DTW asks whether learned speech representations are similar;
- ASR kana agreement asks whether the content is recoverable by a recognizer;
- GOP asks whether acoustic evidence supports the **expected phone rather than competing phones**.

That distinction is exactly why GOP belongs near the center of the clarity/segmental pipeline.

## 2. Classic GOP is useful, but not the end state

Traditional GOP depends on phone segmentation/forced alignment. This creates a circular failure mode for L2 speech: if a learner mispronounces a phone, the aligner may move or distort its boundary, and the GOP score is then computed on the wrong segment.

Cao, Fan, Svendsen and Salvi (Interspeech 2024) explicitly identify this issue and show that CTC models can support GOP-style pronunciation assessment. Their alignment-independent CTC framework also represents deletion, insertion and substitution errors that conventional pre-segmented GOP cannot model naturally. Their best method reports a 29.02% relative improvement over baseline GOP methods on CMU-Kids and SpeechOcean762.

The follow-up “Segmentation-free Goodness of Pronunciation” work introduces:

- **GOP-SA**: self-alignment GOP for CTC models;
- **GOP-SF**: segmentation-free GOP that marginalizes over possible segmentations of the canonical sequence;
- normalization designed to address differences in CTC peakiness.

For our engineering plan, this means MFA/forced alignment should not be a hard prerequisite for the future clarity score. A CTC phone model can directly provide a more robust alignment/scoring path.

## 3. Do not use softmax posterior GOP alone

Recent work gives a strong warning against blindly treating posterior probabilities as well-calibrated pronunciation evidence.

Parikh et al. (Interspeech 2025) compare probability-based and logit-based GOP. They report that logit-based variants improve mispronunciation classification in their L2 English experiments, and maximum-logit GOP aligns best with human perception among tested variants. They recommend hybrid evidence with uncertainty and phone-specific weighting.

Li et al. (BEA 2026) make the CTC-specific issue even clearer: standard CTC is peaky and context-independent, which can make sparse posteriors unstable for GOP. Their context-aware CTC with output-context dependency, label prior and conditional-entropy regularization improves phoneme-level GOPT PCC from 0.612 to 0.641 and increases the correct/mispronounced margin from 0.708 to 0.816 on SpeechOcean762.

Therefore the first Japanese implementation should log **multiple raw features**, not collapse to one formula immediately:

- target-phone mean/max logit;
- best-competitor mean/max logit;
- logit margin;
- target posterior/log posterior;
- posterior GOP-style margin;
- entropy/uncertainty;
- self-aligned CTC path support/duration;
- predicted competitor phone;
- insertion/deletion/substitution evidence where available.

The product can later select or fuse these only after Japanese/L2 benchmarking.

## 4. Japanese phone-CTC backends that are actually available now

### 4.1 Project Beatrice Japanese HuBERT phoneme CTC v4

`prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`

Why it matters:

- Japanese phoneme CTC, not character ASR;
- HuBERT base, 94.4M parameters;
- 16 kHz Transformers-compatible model;
- Apache-2.0 model card;
- phone vocabulary includes Japanese-specific tokens such as `cl`, `N`, `sh`, `ch`, `ts`, palatalized consonants, `pau` and `sil` in the documented family;
- training labels in the model family are derived from Japanese readings/pyopenjtalk-style processing.

This is the best **first deployment/benchmark candidate** because model size is practical for Mac CPU and more realistic for a Hugging Face Space than a 300M+ model.

Caveats:

- the model is trained for native Japanese phone recognition, not L2 pronunciation scoring;
- PER/native ASR quality does not prove correlation with human L2 pronunciation ratings;
- the phone inventory must be audited against our exact pyopenjtalk output (devoiced-vowel conventions, silence, `cl`, compound/palatalized phones);
- v4's own model card mainly documents training changes rather than a full L2 evaluation.

### 4.2 Project Beatrice v5

`prj-beatrice/japanese-hubert-base-phoneme-ctc-v5`

The repository exists and has a 94.4M-parameter checkpoint, but currently exposes no model card that establishes why it is preferable to v4 or provides a clean final evaluation description. Treat it as a **benchmark candidate, not the default** until measured on our panel.

### 4.3 sakasegawa dual CTC Japanese model

`sakasegawa/japanese-wav2vec2-large-hiragana-ctc`

Why it is unusually attractive scientifically:

- 315.6M parameters;
- layer 12 has a dedicated phoneme CTC head (43 classes);
- layer 24 has kana CTC (84 classes);
- the public source exposes raw `phoneme_logits` directly;
- the source phone vocabulary is explicitly pyopenjtalk-native style and includes `A E I N O U`, `cl`, palatalized phones, etc.;
- the model card reports native-Japanese PER of 10.42% on JSUT, 21.43% on JVS parallel100 and 21.87% on a ReazonSpeech test condition;
- Apache-2.0 model/repository metadata.

This is probably the best **scientific comparison backend**, especially because the target phone inventory aligns cleanly with the frontend we already use. It is not the first product default because it is much heavier (~630 MB FP16 checkpoint / 315.6M params).

The model card also flags long-vowel/kana instability. That reinforces a central Japanese design rule: long-vowel correctness must not be delegated to phone posterior alone.

### 4.4 narabas

`darashi/narabas` is an existing Japanese phoneme forced aligner using Wav2Vec2, pyopenjtalk-generated phone sequences and a pretrained Japanese model. Its repository is MIT-licensed and explicitly calls itself experimental. The README reports about 6% naive phoneme error rate on a Common Voice validation setup, while also noting limited hyperparameter exploration.

It is worth adding to the **alignment benchmark**, because it can provide Japanese phone boundaries without bringing in the full MFA stack. It should not be promoted as a pronunciation scorer merely because it aligns phones.

### 4.5 Montreal Forced Aligner

MFA has official Japanese acoustic/dictionary support, and its current documentation includes direct Japanese alignment examples. The 2026 MFA paper evaluates English, Japanese and Korean and reports state-of-the-art or near-state-of-the-art alignment with mean boundary error below 15 ms across four benchmark datasets.

Use MFA as:

- an offline/reference phone-boundary benchmark;
- a way to audit special-mora durations and our own CTC boundaries;
- a possible research aligner.

Do **not** make MFA a hard runtime dependency for clarity. Alignment is not pronunciation correctness, and modern CTC-GOP provides paths that do not need externally fixed phone segments.

## 5. Japanese-specific special morae change the GOP design

Japanese `促音・長音・撥音` cannot be reduced to ordinary phone identity classification.

Kawai & Hirose's Japanese CALL work showed that long vs short vowels can be spectrally very similar while differing critically in duration; analogous duration distinctions matter for mora nasals and obstruent/geminate contrasts. Their system explicitly measured phone durations and linked those durations to native-listener perceptual confusability.

Therefore:

- `cl` posterior/GOP is useful evidence for 促音, but closure/timing duration remains essential;
- `N` posterior/GOP is useful evidence for 撥音, but context-dependent realization makes a single categorical phone score insufficient;
- 長音 is primarily a mora/duration contrast and may not correspond to a distinct long-vowel phone token in every model;
- special-mora diagnostics should fuse **phone evidence + relative duration/timing**, and remain primarily in the `リズム` diagnostic family rather than being swallowed by `明瞭さ`.

The broader Japanese timing literature also rejects a simplistic “all moras must be equal duration” rule in spontaneous speech. So phone durations should be interpreted relationally/contextually rather than by a universal fixed duration threshold.

## 6. Relationship with the existing WavLM work

The 2026 McIntosh et al. study is directly relevant to this project because it evaluates English and Japanese L2 speech and separates phone, rhythm and intonation scoring using WavLM/DTW-style comparison. It supports WavLM as a strong reference-comparison signal for phone and rhythm, while intonation remains harder.

Our own existing v3.2 report already found the expected pooled ordering `native LOO < learner << wrong target` for multi-reference WavLM, but also found material speaker/channel effects and a target (`バグ`) with weak learner/native separation. Therefore WavLM remains valuable, but should not be mistaken for pure phone correctness.

Recommended complementarity:

- **CTC-GOP**: canonical-phone support / likely substitution diagnostics;
- **WavLM multi-reference**: broader reference-relative pronunciation similarity;
- **ASR kana agreement**: machine intelligibility/content recoverability;
- **duration/timing**: special mora + rhythm;
- **F0**: intonation.

A future clarity score can combine GOP + WavLM + ASR, but only after benchmark/calibration. Do not average their `/100` values mechanically.

## 7. Fixed reading versus free speaking

### Fixed reading

This is the easiest and strongest GOP setting:

`target text -> pyopenjtalk -> canonical phones -> phone-CTC logits -> CTC-GOP`

No ASR transcript is needed to define the phone target. Content verification remains useful as a guard against reading a different sentence.

### Free speaking

The architecture is also attractive once the transcript is confirmed:

`language-aware ASR -> user-confirmed Japanese transcript -> pyopenjtalk -> canonical phones -> phone-CTC GOP against the original audio`

This is strategically important because phone-level clarity no longer needs a TTS pseudo-reference. A TTS reference can still be generated for playback/practice, but it does not need to define pronunciation correctness.

This should substantially clean up the current weak-reference semantics.

## 8. Proposed implementation sequence

### Stage A — shadow GOP adapter (first)

Implement an optional `phoneme_gop.py` that is lazy and disabled in ordinary tests.

Backend interface:

- `beatrice_hubert_phone_ctc_v4` first;
- `sakasegawa_dual_ctc_phone_head` second;
- future backends can share the same output schema.

Output per canonical phone:

- canonical phone;
- phone index / mora association;
- CTC span/path support;
- target logit statistics;
- best competitor and competitor logit;
- logit margin;
- target posterior/log-posterior;
- posterior GOP margin;
- entropy;
- duration;
- confidence/evidence tier;
- insertion/deletion/substitution diagnostics if derivable.

Utterance summary should preserve distributions (median, lower quantile, worst-supported phones, proportion below candidate thresholds) rather than only a mean.

No `/100` product mapping in Stage A.

### Stage B — compare scoring definitions

Benchmark at least:

1. posterior GOP;
2. mean-logit margin;
3. max-logit GOP/margin;
4. CTC self-aligned GOP (GOP-SA style);
5. if implementation cost is reasonable, GOP-SF/segmentation-free score.

Do not assume the classic posterior formula wins.

### Stage C — Japanese error-aware competitors

Only after Stage B, add restricted competitor sets informed by Japanese phonology and observed learner confusions. The 2025 substitution-aware GOP work suggests this can improve alignment-free efficiency, but the English-specific confusion sets cannot simply be copied into Japanese.

Potential Japanese clusters to investigate empirically include place/manner neighbors and L1-sensitive confusions, but these must be derived from actual Japanese learner data rather than hard-coded from intuition.

## 9. Benchmark gates before product promotion

Use the existing real/synthetic panel, but add phone-level outputs. Minimum comparisons:

- native correct target;
- learner correct target;
- wrong Japanese target;
- same target under gain/noise/RIR/codec/bandlimit;
- global speed perturbation;
- inserted pause;
- consonant attenuation/local deletion/vowel spectral perturbation;
- 促音/長音/撥音 target words;
- different native speakers.

A useful GOP signal should not merely separate “different sentence” from “same sentence”. More important tests are:

- local phone edit causes a stronger phone-local GOP change than an innocuous gain change;
- correct native variants are not punished simply for speaker identity;
- channel corruption does not dominate learner/native separation;
- phone-local evidence points to the edited/mispronounced phone rather than shifting errors arbitrarily;
- speed/pause manipulations primarily affect fluency/rhythm, not clarity GOP, unless they actually destroy segment realization;
- `cl`, `N`, vowel-duration cases cooperate with special-mora timing evidence instead of contradicting it.

Do not promote a backend just because native PER is low.

## 10. Licensing / deployment note

The candidate model cards/repositories above label their model/code artifacts Apache-2.0 or MIT. However, both major Japanese phone-model families were trained using ReazonSpeech-derived data. Reazon's official site distinguishes Apache-2.0 ASR models from the corpus itself, and the corpus is distributed under CDLA-Sharing-1.0 with an additional usage condition tied to Japanese Copyright Act Article 30-4.

Engineering implication:

- do not download/use the ReazonSpeech corpus itself for product retraining without a separate legal/data-governance review;
- record model-card license, upstream base-model license, training-data provenance and model artifact hash for every production candidate;
- benchmark published weights first;
- “free to download” must not be treated as equivalent to “unrestricted training data”.

This report does not make a legal conclusion about downstream model-weight use.

## 11. Decision matrix

| Candidate | Phone logits | Japanese-specific | pyopenjtalk compatibility | Runtime size | Main role | Current verdict |
|---|---|---|---|---:|---|---|
| Beatrice HuBERT phoneme CTC v4 | Yes | Yes | Close; adapter audit needed | 94.4M / ~378MB F32 | first GOP backend | **IMPLEMENT SHADOW FIRST** |
| Beatrice v5 | Yes | Yes | audit needed | 94.4M / ~378MB F32 | comparison | BENCHMARK AFTER v4 |
| sakasegawa dual CTC | Yes, dedicated phone head | Yes | Very strong / pyopenjtalk-style vocab | 315.6M / ~630MB FP16 | scientific comparison / possible higher-quality backend | **IMPLEMENT SECOND** |
| narabas | alignment output | Yes | pyopenjtalk phones | large Wav2Vec2 lineage | phone boundary benchmark | BENCHMARK, NOT SCORE |
| MFA Japanese | alignment/posteriors in aligner stack | Yes | separate phone mapping | offline stack | boundary gold/reference | BENCHMARK, NOT C-END DEFAULT |
| Existing WavLM-large | representation, not phone logits | multilingual incl. Japanese evidence | N/A | large | holistic phonetic similarity | KEEP COMPLEMENTARY |
| Whisper/faster-whisper | text ASR | Japanese supported | kana via frontend | existing | content/intelligibility | KEEP, NOT PHONE SCORE |

## 12. Final recommendation

The next scientific/engineering milestone should be **Japanese CTC-GOP shadow v1**, not further hand-tuning of the current ASR-based clarity proxy.

The first implementation should use Beatrice v4 for deployability and expose posterior + logit + uncertainty features. The second should benchmark sakasegawa's dedicated phone head because its pyopenjtalk-style inventory and dual CTC architecture make it an unusually good fit. After that, implement/compare GOP-SA and, if tractable, segmentation-free GOP-SF.

MFA/narabas should be used to audit boundaries, not to define pronunciation correctness. WavLM should stay as an independent reference-similarity signal. Special morae must keep their duration/timing path.

Only after this benchmark should the C-end `明瞭さ` evidence ladder be reordered so GOP becomes the primary high-tier signal and ASR/acoustic similarity becomes fallback evidence.

## Primary references / official resources

- Witt, S. M. & Young, S. J. (1998). *Performance measures for phone-level pronunciation teaching in CALL*. STiLL 1998.
- Kaldi official `gop_speechocean762` recipe and `compute-gop` implementation.
- Cao, X., Fan, Z., Svendsen, T., & Salvi, G. (2024). *A Framework for Phoneme-Level Pronunciation Assessment Using CTC*. Interspeech 2024. DOI: 10.21437/Interspeech.2024-1785.
- Cao, X., Fan, Z., Svendsen, T., & Salvi, G. (2025/2026). *Segmentation-free Goodness of Pronunciation*. arXiv:2507.16838; later journal publication.
- Parikh, A. K. et al. (2025). *Evaluating Logit-Based GOP Scores for Mispronunciation Detection*. Interspeech 2025. DOI: 10.21437/Interspeech.2025-1012.
- Parikh, A. K. et al. (2025). *Enhancing GOP in CTC-Based Mispronunciation Detection with Phonological Knowledge*. Interspeech 2025. DOI: 10.21437/Interspeech.2025-829.
- Li, J.-T. et al. (2026). *Investigating Context-aware CTC for Pronunciation Assessment: Mitigating Peaky Behavior and Context Independency Assumption*. BEA 2026. DOI: 10.18653/v1/2026.bea-1.3.
- Kawai, G. & Hirose, K. (1998). *A CALL system using speech recognition to teach the pronunciation of Japanese tokushuhaku*. STiLL 1998.
- Warner, N. & Arai, T. (2001). *The role of the mora in the timing of spontaneous Japanese speech*. JASA 109(3), 1144–1156.
- McIntosh, S. et al. (2026). *Self-supervised Speech Comparison for L2 Phone, Rhythm, and Intonation Scoring*. arXiv:2607.13721.
- Official model cards/repos: Project Beatrice Japanese HuBERT phoneme CTC v4/v5; sakasegawa Japanese dual CTC; darashi/narabas; Montreal Forced Aligner Japanese models/docs; ReazonSpeech licensing page.

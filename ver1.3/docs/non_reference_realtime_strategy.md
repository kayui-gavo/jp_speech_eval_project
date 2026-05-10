# Non-Reference Realtime Strategy

This note is for the speech evaluation module only. It separates what can be
done in realtime without a target sentence from what still needs ASR, a
reference, or a trained pronunciation model.

## 1. Streaming ASR is useful, but it is not pronunciation assessment

Streaming ASR can help the system form a live content hypothesis:

- what the user probably said
- when a phrase probably ended
- whether the current utterance is Japanese-like enough to continue analysis
- what kana/mora sequence can be generated after a stable partial transcript

But ASR is trained to recover words, not to grade learner pronunciation.
It may normalize or "repair" learner speech. If the user says a wrong or
unclear sound and ASR still outputs the intended word, the ASR result can hide
the pronunciation problem.

Therefore, the product should use streaming ASR as an interface signal, not as
the final correctness judge.

Recommended realtime split:

- 0-300 ms: wav-only acoustic monitor, no transcript dependency
- 300-900 ms: partial ASR only for content hypothesis and phrase detection
- after a phrase is stable: generated kana/mora target for light diagnosis
- after speech end: heavier pseudo-reference or reference-based analysis

## 2. The true non-reference mode has a hard boundary

If the input is only `user_wav` and we do not use ASR transcript, target text,
or TTS reference, the system cannot know what the user intended to say.

It can output:

- endpointing and recording quality
- voiced ratio
- pause ratio
- F0 range after speaker normalization
- energy stability
- speech rhythm risk
- broad pronunciation risk proxy
- reliability

It cannot output:

- kana correctness
- word correctness
- long vowel / sokuon / nasal correctness for a specific target
- pitch accent correctness for a specific word
- "you read the sentence correctly"

This mode should be named `reference_free_acoustic`, not "pronunciation
correctness".

## 3. ASR pseudo-reference mode is the practical bridge

ASR-generated reference mode should be named `asr_pseudo_reference` or
`transcript_generated_reference`.

Flow:

1. user wav or streaming chunk
2. ASR transcript
3. transcript to kana/mora/accent candidates by pyopenjtalk/OJAD/dictionary
4. optional TTS/generated reference
5. contour/rhythm/alignment analysis

This is not pure non-reference, because the transcript becomes a generated
reference. It is useful for free-speaking practice, but all output must say
"based on ASR transcript". If ASR confidence is low or the transcript changes
frequently, mora-level feedback should be suppressed.

## 4. Minematsu / OJAD ideas that are directly useful

The most useful idea is not "use one absolute pitch target". The useful idea is:
compare structures after removing speaker-specific traits.

For this project, that means:

- compare speaker-normalized log-F0 contour, not raw Hz
- compare rising/falling/flat transitions, not absolute pitch height
- separate linguistic prosody from emotion/style
- use visual feedback for accent and intonation
- treat generated accent labels as hypotheses unless backed by dictionary/manual data

OJAD is useful as a pitch accent and phrase-intonation reference source. It is
not a complete pronunciation scorer. Its sentence-level text search is also
automatic and can be wrong, so it should be marked as dictionary/tool-derived
evidence with reliability.

## 5. Product modes to keep

`reference_fixed_sentence`

- best for textbook sentences, shadowing, and mora-level correction
- can output pronunciation/prosody correctness if alignment is reliable
- requires target text and preferably curated reference/accent labels

`asr_pseudo_reference`

- best for free speech after ASR transcript stabilizes
- can output "based on what ASR heard" diagnosis
- should not claim strict correctness if ASR may have repaired the content

`reference_free_acoustic`

- best for realtime lightweight feedback while the user is speaking
- can output recording quality, pauses, voicing, pitch-range/style proxies
- cannot judge specific kana correctness

`transcript_assisted_light`

- future mode for user wav + transcript, without TTS reference
- can estimate mora count, articulation rate, pause risk, final intonation
- lighter and more stable than generating TTS every time

`gop_ctc_future`

- future pronunciation evidence layer
- needed for robust phoneme/kana correctness and learner-specific error types

## 6. Research/product roadmap

Near-term:

- cache ASR models in-process for lower repeated-call latency
- keep acoustic-only monitor active during waiting/speaking
- run ASR only on phrase-size buffers or after endpoint `maybe_ending`
- mark all ASR-derived targets as pseudo-reference
- avoid running ASR twice in `asr_pseudo_reference`; once the transcript is the
  pseudo-reference, content matching should not call ASR again
- use `transcript_assisted_light` as the default low-latency free-speaking path
  when feedback must appear within about 2 seconds
- treat TTS-generated reference audio as an optional debug aid, not as a native
  standard pronunciation
- log speaker-normalized F0 movement, mora rate, special-mora density, and
  compression-risk flags as trainable structural features. These are closer to
  Minematsu-style structural comparison than raw Hz or speaker identity.

Next:

- add OJAD/accent dictionary source for pitch targets
- add transcript-assisted light diagnosis
- train a fast regression/classification model on acoustic proxies and teacher labels
- improve mora timing with transcript-conditioned duration priors and reliability
  gates before moving to heavier aligners

Later:

- add CTC/GOP posterior evidence, preferably variants that can account for
  deletion/insertion/substitution errors instead of depending only on brittle
  forced-alignment segments
- train learner speech calibrated models
- evaluate per-L1 accent patterns separately

## 7. Latency target

For product interaction, the free-speaking path should aim for:

- under 300 ms for endpoint state / volume / basic acoustic monitor
- under 2 seconds after stop for transcript-assisted feedback
- sentence-final detailed reference or pseudo-reference feedback can be slower,
  but should be clearly labeled as detailed analysis

The current main bottlenecks are:

- ASR inference
- browser-side WAV encoding/upload/wait overhead
- generated TTS reference and MFCC/F0 cache build
- DTW / mora-boundary estimation

The fastest reliable route is not to generate a TTS reference every time. For
free speech, use ASR transcript to estimate kana/mora count, then score acoustic
and structural proxies conservatively.

## 8. Sources

- OJAD official site: https://www.gavo.t.u-tokyo.ac.jp/ojad/eng/pages/home
- Nakamura et al., "Development of a web framework for teaching and learning
  Japanese prosody: OJAD", Interspeech 2013:
  https://www.isca-archive.org/interspeech_2013/nakamura13_interspeech.html
- Hirano et al., "OJAD: a free online accent and intonation dictionary for
  teachers and learners of Japanese", SLaTE 2013:
  https://www.isca-archive.org/slate_2013/hirano13_slate.html
- Minematsu and Suzuki, "Structure-based pronunciation assessment", SLaTE 2009:
  https://www.isca-archive.org/slate_2009/minematsu09_slate.html
- Minematsu, "音声分析・合成・認識技術を用いた多様な外国語教育支援",
  日本音響学会誌 2018:
  https://www.jstage.jst.go.jp/article/jasj/74/9/74_525/_article/-char/ja/
- Cao et al., "A Framework for Phoneme-Level Pronunciation Assessment Using
  CTC", Interspeech 2024:
  https://www.isca-archive.org/interspeech_2024/cao24b_interspeech.html

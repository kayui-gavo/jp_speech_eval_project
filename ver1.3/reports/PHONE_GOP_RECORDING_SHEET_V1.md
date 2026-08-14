# Japanese phone-GOP quick recording sheet v1

Use with `data/phone_gop_manual_validation_manifest_v2.csv`.

This sheet is deliberately practical: read from top to bottom and save each WAV with the shown `clip_id`. Normal takes come before intentional-error takes so the error instruction does not contaminate the clean pronunciation.

## Before recording

- Same room, same microphone, same approximate distance.
- Normal conversational loudness.
- Do not imitate a TTS voice.
- For N2, speak naturally again rather than trying to reproduce N1 acoustically.
- For an error take, change only the requested property as much as possible.
- If you cannot actually hear the intended error when listening back, keep the file but later mark `realized_as_intended=uncertain/no`; do not treat the instruction itself as ground truth.

## A. 促音

- [ ] `GEM_KAKKO_N1.wav` — それは、かっこです。 — natural
- [ ] `GEM_KAKKO_N2.wav` — それは、かっこです。 — natural again
- [ ] `GEM_KAKKO_E.wav` — target text remains それは、かっこです。 — actually say それは、かこです。 Remove the extra closure; do not merely speed up the whole word.

## B. 長音

- [ ] `LV_OBAASAN_N1.wav` — それは、おばあさんです。 — natural
- [ ] `LV_OBAASAN_N2.wav` — それは、おばあさんです。 — natural again
- [ ] `LV_OBAASAN_E.wav` — target text remains それは、おばあさんです。 — actually say それは、おばさんです。 Remove only the long-vowel mora.

## C. 撥音

- [ ] `N_MINNA_N1.wav` — それは、みんなです。 — natural
- [ ] `N_MINNA_N2.wav` — それは、みんなです。 — natural again
- [ ] `N_MINNA_E.wav` — target text remains それは、みんなです。 — actually say それは、みなです。 Remove the moraic nasal timing.

## D. 拗音 split

- [ ] `YO_KYAKU_N1.wav` — それは、きゃくです。 — natural, きゃ as one mora
- [ ] `YO_KYAKU_N2.wav` — それは、きゃくです。 — natural again
- [ ] `YO_KYAKU_E.wav` — target remains きゃく — actually say きやく, with き・や as two timing units

## E. Vowel /u/→/o/

- [ ] `V_YUME_N1.wav` — それは、ゆめです。 — natural
- [ ] `V_YUME_N2.wav` — それは、ゆめです。 — natural again
- [ ] `V_YUME_E.wav` — target remains ゆめ — actually say よめ. Keep /y/ and change the vowel nucleus only.

## F. Japanese ふ /f/-like versus /h/-like

- [ ] `F_FUNE_N1.wav` — それは、ふねです。 — natural Japanese ふ
- [ ] `F_FUNE_N2.wav` — それは、ふねです。 — natural again
- [ ] `F_FUNE_E.wav` — target remains ふね — use a throat/glottal h-like consonant like は, deliberately reducing the lip-frication quality of normal Japanese ふ. If you cannot produce/hear the contrast reliably, mark this item uncertain later.

## G. /ts/→/s/

- [ ] `TS_TSUKI_N1.wav` — それは、つきです。 — natural
- [ ] `TS_TSUKI_N2.wav` — それは、つきです。 — natural again
- [ ] `TS_TSUKI_E.wav` — target remains つき — actually say すき

## H. /sh/→/s/

- [ ] `SH_SUSHI_N1.wav` — それは、すしです。 — natural
- [ ] `SH_SUSHI_N2.wav` — それは、すしです。 — natural again
- [ ] `SH_SUSHI_E.wav` — target remains すし — keep the following /i/, but produce the consonant of し with an s-like configuration rather than the normal Japanese sh-like articulation. If playback does not reveal a clear difference, do not treat it as a positive error sample.

## I. Voicing /b/→/p/

- [ ] `VOICE_BUS_N1.wav` — それは、バスです。 — natural
- [ ] `VOICE_BUS_N2.wav` — それは、バスです。 — natural again
- [ ] `VOICE_BUS_E.wav` — target remains バス — actually say パス

## J. Voicing /g/→/k/

- [ ] `VOICE_KAGI_N1.wav` — それは、かぎです。 — natural
- [ ] `VOICE_KAGI_N2.wav` — それは、かぎです。 — natural again
- [ ] `VOICE_KAGI_E.wav` — target remains かぎ — actually say かき

## K. Four-dimension orthogonality controls

The target sentence is always `ラーメンをください。`. Keep consonants/vowels as stable as possible.

- [ ] `CTRL_RAMEN_NORMAL.wav` — natural
- [ ] `CTRL_RAMEN_FAST.wav` — roughly 15–25% faster overall; do not intentionally swallow phones
- [ ] `CTRL_RAMEN_SLOW.wav` — roughly 15–25% slower overall; do not selectively exaggerate the long vowel
- [ ] `CTRL_RAMEN_PAUSE.wav` — normal local rate, but insert roughly 0.6–0.8 s silence between ラーメンを and ください
- [ ] `CTRL_RAMEN_FLAT.wav` — normal segments/rate, intentionally flatter pitch movement
- [ ] `CTRL_RAMEN_EXAG.wav` — normal segments/rate, deliberately exaggerated pitch rises/falls

Expected direction, qualitatively:

- pause should mainly hit fluency;
- fast/slow should hit rhythm/fluency more than clarity;
- flat/exaggerated pitch should hit intonation more than clarity;
- GOP phone evidence should not collapse for a pure pitch manipulation.

## L. Legitimate Japanese variation control

- [ ] `DEV_SUKIDESU_NAT.wav` — すきです。 — say it naturally; do not force high vowels to stay fully voiced
- [ ] `DEV_SUKIDESU_FULL.wav` — すきです。 — comparison take with unusually careful/full vowel voicing

Both should remain broadly acceptable. This is a negative control against treating native-like high-vowel devoicing as a severe phone error.

## After quick recording

Do **not** immediately label every E file as a successful error. First randomize filenames/order for listening inspection, then use `data/phone_gop_manual_inspection_template_v1.csv`.

For each E take, first listen blind and record what you actually hear. Only afterward reveal the intended manipulation and mark `realized_as_intended=yes/no/uncertain`.

The quick sheet contains 38 recordings. The extended manifest adds Chinese-learner-motivated sokuon patterns, three /N/ contexts, additional palatalized-mora cases and an optional /r/ distortion control.

# Phone-GOP Stage-0 empirical audit v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Product impact: **NONE**
Human-recording gate: **BLOCKED**

## Scope

This report records the first complete automatic heavy-preflight artifact review
before any new human GOP recording. It deliberately separates observations that
are valid engineering evidence from observations invalidated by a target-side
bug discovered during the audit.

The reviewed heavy run was GitHub Actions run `31868538253`. Its unit-test job
passed with **301 tests / 6 existing warnings**, and the full model-preflight job
completed successfully. The branch subsequently received additional tests and
at the time of the target-reading fix ordinary CI passed **304 tests / 6 existing
warnings**.

## 1. Controlled local phone edits: useful automatic evidence

The ephemeral OpenJTalk control battery evaluated seven deliberately local
contrasts:

- `b -> p`
- `g -> k`
- `ts -> s`
- `sh -> s`
- `ch -> sh`
- `cl` deletion
- `N` deletion

Result:

- target-specific LPR moved in the expected direction in **7 / 7** cases;
- **6 / 7** also crossed from positive on the correct synthetic utterance to
  negative on the manipulated utterance;
- all intended substitution candidates were present in the restricted search.

The one non-sign-flip case was `N` deletion: the deletion LPR fell strongly but
remained positive. More importantly, the `ts -> s` control already had a
negative `ts`-vs-`s` LPR on the *correct* synthetic `つきです` utterance, even
though the erroneous `すきです` realization made it much more negative.

**Decision:** a universal rule such as `LPR < 0 => learner error` is rejected.
The directional paired contrast is useful as Stage-0 implementation evidence,
but learner decisions require criterion calibration and phone-specific behavior.

## 2. Public UME-JRF learner-domain samples: promising but unlabeled

The official NII-SRC public UME-JRF listening samples were downloaded only
inside CI and deleted before artifact upload. The public page identifies them as
utterances by native speakers of Chinese, but does not expose the four-teacher
rating lists for these individual sample WAVs. They are therefore real
Japanese-L2 **domain anchors**, not correctness labels.

Descriptive restricted noncanonical-outscore rates on the five public samples:

- `A1_001`: 0.1154
- `B1_001`: 0.0000
- `C1_001`: 0.5000
- `D1_001` (`じぶつ`): 0.1667
- `D1_002` (`じんぶつ`): 0.0000

These rates must not be read as learner error rates.

The published `じぶつ / じんぶつ` minimal pair is nevertheless a valuable
same-speaker/domain sanity contrast. On each waveform, the sequence associated
with the page's own prompt was preferred over the partner:

- `D1_001 じぶつ` vs `じんぶつ`: own-target minus partner log posterior
  **+34.73**;
- `D1_002 じんぶつ` vs `じぶつ`: own-target minus partner log posterior
  **+84.20**.

The two canonical phone strings differ by exactly one mora-nasal `N`, verified
by repository test. This supports sensitivity to `N` presence/absence on these
two real L2 examples, but still does not establish an educational pronunciation
threshold.

## 3. CTC posterior regimes differ strongly across backbones

On the same bundled `ラーメンをください` audio, posterior diagnostics showed:

| Backbone | mean top-1 posterior | blank top-1 fraction | mean phone mass | normalized full entropy |
|---|---:|---:|---:|---:|
| Beatrice | 0.943 | 0.786 | 0.178 | 0.0378 |
| DistilHuBERT dual CTC | 0.989 | 0.859 | 0.127 | 0.0142 |
| WavLM dual CTC | 0.997 | 0.866 | 0.128 | 0.00413 |

All three are substantially peaky, with DistilHuBERT/WavLM especially so. This
is a model-property diagnostic, not a quality ranking. The result reinforces the
decision to keep logit features, posterior/LPR features, entropy/uncertainty and
alignment-free features separate until Japanese-L2 criterion validation.

Correct-target vs unrelated wrong-target sequence log-posterior gaps on the
same bundled audio were large for all three backbones:

- Beatrice: **+331.48**
- DistilHuBERT: **+391.57**
- WavLM: **+406.53**

This is strong target-specificity evidence, not phone-level learner validity.

## 4. Critical audit finding: the first JVS native false-alarm result was invalid

The first restricted JVS artifact appeared to show the exact same 9 negative
phone positions out of 77 for all three native speakers (rate 0.11688). The
identical block immediately suggested a target-side problem rather than three
speakers independently making the same local errors.

Inspection of the target phone sequence found the cause. Automatic surface-kanji
G2P analyzed the `明王` portion as phones corresponding to an `あきら...` reading,
while the sentence means and is spoken as `みょうおう`. The affected phone block
lined up with the repeated native "false alarms".

Therefore:

- the old JVS `9 / 77` value is **withdrawn** and must not be cited as a native
  false-alarm rate;
- the old three-backbone JVS target-specificity artifact is also superseded for
  phone-level interpretation because it used the same bad canonical sequence;
- transport/source provenance was fine; the failure was target pronunciation
  provenance.

This is exactly the type of bug Stage-0 was intended to catch before human
recording.

## 5. Target reading / phone evidence has been hardened

The JVS download manifest now stores a reviewed reading override:

`また、とうじのように、ごだいみょうおうとよばれる、しゅようなみょうおうのちゅうおうにはいされることもおおい。`

and explicitly marks automatic surface-only G2P as unsafe for that anchor.

The target frontend was also corrected more generally: after a reading is
resolved (automatic, verified, or manual), the phone sequence is now generated
from the **resolved reading**, rather than independently re-analyzing surface
kanji. This prevents a verified/manual kana target from disagreeing with its own
phone sequence.

A regression test explicitly checks `明王` with `みょうおう` override and the
full JVS sentence.

## 6. Special-mora construct separation was tightened

`N` and `cl` remain legitimate canonical phone tokens and sequence-level
research evidence. They are no longer allowed as generic ordinary segmental
competitors in the clarity search space. This aligns the implementation with the
product construct design:

- ordinary segmental clarity: ordinary phone competitors;
- `N`, `cl`, long-vowel behavior: dedicated special-mora / timing evidence,
  with CTC sequence evidence as a supporting signal.

No special-mora phone token was deleted from target representations.

## 7. What remains blocked

This automatic evidence is now much stronger, but it still does not authorize a
learner-facing GOP `/100` or the user's recording time.

Before human-recording promotion, the corrected JVS heavy rerun must confirm:

1. reviewed target reading is actually used by all three backbones;
2. the repeated old `明王` false-alarm block disappears;
3. JVS native restricted behavior remains inspectable after special-mora
   separation;
4. UME public learner-domain and controlled TTS preflights remain available;
5. hybrid/logit/alignment-free/peakiness bundles remain finite and provenance-
   consistent.

After that, the next scientifically meaningful step is **labeled criterion
validation** (full UME-JRF or equivalent expert phone labels), not immediate
consumer score mapping.

Current product policy remains unchanged:

- `score_mapped = false` for GOP research evidence;
- `product_calibrated = false`;
- C-end four-score fallback is untouched;
- no new human GOP recording is requested.

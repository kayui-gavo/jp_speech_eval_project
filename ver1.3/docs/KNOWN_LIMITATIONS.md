# Known Limitations

- The practice score is not calibrated to human proficiency ratings and must
  not be presented as an educational-measurement score.
- Fixed-reference acoustic similarity does not establish phone correctness.
- ASR transcript sanity is a lightweight guard, not a semantic Japanese
  acceptability model; false acceptance and false rejection remain possible.
- A broad mismatch fallback evaluates the recorded Japanese generally and
  cannot claim that the displayed sentence was read correctly.
- Lexical pitch targets generated only by OpenJTalk are weak targets. Strong
  accent feedback requires manually verified or OJAD-reviewed targets.
- Special-mora v2, phrase-intonation v1, accent-nucleus, and SSL pronunciation
  outputs are shadow evidence only. They lack learner-error calibration.
- The default SSL checkpoint is large and optional. Deployments must budget
  model storage and latency before enabling it; ordinary tests never download
  the checkpoint.
- Multi-native-reference SSL aggregation is represented in the interface but
  still requires a reference panel and calibration data.
- Moderate noise is handled by confidence/detail degradation, but severe noise
  can still make audio unscorable.

## Calibration data needed next

Use target-balanced, speaker-disjoint fixed-reading data with several native
references per sentence, real Japanese learners, human intelligibility and
pronunciation ratings, verified accent nuclei, annotated special-mora errors,
and controlled channel/noise pairs. Keep calibration and held-out evaluation
strictly separated and report target/speaker/channel subgroups.

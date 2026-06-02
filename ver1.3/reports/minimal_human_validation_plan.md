# Minimal human validation plan

Goal: quick demo safety check, not a formal experiment.

## Reviewer

- myself first
- optionally 1 Japanese native speaker, Japanese teacher, or speech researcher

## Items

- fixed-reference outputs: 10-20 examples
- special mora manual inspection: use the existing manual review pack
- ASR+Kanade flow: 3-5 examples

## Annotation fields

- intelligibility
- naturalness
- communication impact
- should_feedback
- wording_ok

For special mora, keep using the v2 distinction:

- audible variation does not automatically mean error
- natural / acceptable variation should not become correction
- near-boundary examples should normally pass

## Expected time

- self-check: 1-2 hours
- external quick check: 30-60 minutes

## Decision rule

- If feedback seems unsafe, keep it debug/shadow.
- If long_vowel / moraic_nasal tips are accepted, allow limited candidate only.
- If a reviewer marks wording as too strong, rewrite it before demo.
- If Kanade playback confuses scoring, make the mode notice more visible.

## Deliverable

After the check, summarize:

- examples that should not show user-facing feedback
- confusing wording
- ASR confirmation mistakes
- Kanade playback comments
- any native-like variation that the system nearly over-corrected

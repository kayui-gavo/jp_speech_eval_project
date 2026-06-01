# ASR-confirmed weak-reference flow

ASR-generated reference is a weak-reference practice mode. It is useful for free speech, but it is less reliable than fixed-reference because the target text is inferred from ASR.

## State machine

1. `user_records_free_speech`
2. `asr_transcript_generated`
3. `user_confirm_required`
4. `user_confirms_or_edits_text`
5. `pseudo_reference_generated`
6. `weak_reference_evaluation_runs`
7. `UserFacingResult` returns low/medium confidence or conservative pass

## Required confirmation

ASR raw text must not become a scoring reference directly.

Before evaluation, UI must show:

- candidate text
- editable text box
- confirm button

Only `user_confirmed_text` becomes the weak target.

## Output policy

If the user has not confirmed or edited the ASR text:

- do not output pronunciation correctness
- return confirmation-required / debug-only style state
- do not generate a scoring reference

After confirmation:

- rhythm, fluency, and content-level light feedback are allowed
- pitch feedback is suppressed unless the target is verified
- strong special-mora correction is suppressed by default
- result must keep a weak-reference notice visible

## Limitation

This mode is for practice support. It should not be presented as strict pronunciation assessment.

# Alignment resource audit

This report checks whether local datasets already contain phone/mora alignment labels before running MFA.

## JVS
- exists: True
- transcripts/text files: 803
- phone labels: 25468
- TextGrid files: 0
- duration labels: 1

## JANON
- exists: True
- transcript manifest: 1
- phone labels: 0
- TextGrid files: 0
- duration labels: 0

## Project
- TextGrid parser available: True
- alignment cache dirs: []
- MFA command: not found

## Conclusion
- Existing JVS labels may be usable. Prefer existing_label adapter first.
- MFA is not installed in the current environment, so MFA alignment should be reported as skipped.

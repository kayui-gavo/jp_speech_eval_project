# MFA Alignment Setup

This project treats MFA as an optional offline calibration backend. It is not a
runtime dependency for the consumer demo.

## Install MFA

Use the official Montreal Forced Aligner installation guide for your platform.
One common local setup is:

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa
```

Then verify:

```bash
mfa version
```

## Download Japanese models

MFA needs both a Japanese acoustic model and dictionary. Depending on the MFA
version, the model names may be:

```bash
mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
```

If those names are unavailable, check:

```bash
mfa model list acoustic
mfa model list dictionary
```

## Prepare a small JVS subset

Create a corpus directory containing matching `.wav` and `.txt` files:

```text
corpus/
  VOICEACTRESS100_001.wav
  VOICEACTRESS100_001.txt
```

The text file should contain the Japanese transcript for that utterance.

## Run MFA

```bash
mfa align corpus japanese_mfa japanese_mfa outputs/mfa_jvs_subset --clean
```

The output should contain TextGrid files. Put or copy them into an alignment
cache directory and run calibration with:

```bash
python scripts/run_calibration_snapshot.py \
  --focus-special-mora \
  --alignment-backend mfa \
  --alignment-cache-dir outputs/mfa_jvs_subset
```

## Graceful fallback

If `mfa` is not installed, calibration must not crash. The script will mark MFA
as skipped and keep `reliable_count=0` for threshold generation.

## Common failure causes

- Missing Japanese dictionary/acoustic model.
- Transcript normalization mismatch.
- Wav/text basename mismatch.
- Very short utterances.
- TextGrid tier name not recognized as `phones`, `phone`, or `segments`.

When MFA fails, do not update special-mora thresholds from equal fallback
alignment. Equal fallback is useful for coverage accounting only.

# Alignment backend comparison

This report compares offline alignment evidence backends used for calibration. Equal/MFCC-DTW fallback can support coarse debugging but must not generate special-mora thresholds.

- existing_label: n=17, success_rate=1.0, fallback_rate=0.0, usable_special_mora_rate=1.0, example_failure=
- mfcc_dtw: n=20, success_rate=0.0, fallback_rate=0.0, usable_special_mora_rate=0.0, example_failure=not reliable enough for special mora threshold calibration

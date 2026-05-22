from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.app_core.calibration import calibrate_from_manifest, read_calibration_manifest
from jp_speech_eval.app_core.user_profile import save_user_profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a lightweight user voice profile from calibration recordings.",
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--manifest", required=True, help="CSV with text,audio_path[,cache_path].")
    parser.add_argument("--out", default="outputs/user_profiles/profile.json")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    rows = read_calibration_manifest(args.manifest)
    if not rows:
        parser.error("Manifest must contain rows with text and audio_path.")

    profile = calibrate_from_manifest(
        user_id=args.user_id,
        rows=rows,
        scoring_config_path=args.config,
        sample_rate=args.sr,
    )
    save_user_profile(profile, args.out)

    summary = {
        "user_id": profile.user_id,
        "sample_count": len(profile.calibration_samples),
        "profile_path": args.out,
        "baseline_scores": profile.baseline_scores,
        "voice_baseline": {
            "f0_median_hz": profile.f0_median_hz,
            "f0_range_log": profile.f0_range_log,
            "mora_rate_avg": profile.mora_rate_avg,
            "avg_mora_duration_sec": profile.avg_mora_duration_sec,
            "pause_ratio_avg": profile.pause_ratio_avg,
        },
        "common_issues": profile.common_issues,
        "note": "This profile is for progress feedback only; it does not replace standard Japanese targets.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

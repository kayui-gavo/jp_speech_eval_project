#!/usr/bin/env python3
"""Execute the v10 readiness workflow only from a verified v11 freeze.

The held set and promotion protocols are read from the freeze artifact.  Any
manifest, protocol, or audio-byte drift fails before evaluation starts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from free_speech_confirmatory_freeze_v11 import verify_freeze
from run_free_speech_promotion_readiness_v10 import run as run_v10


RUN_SCHEMA = "free_speech_confirmatory_run_v11"


def _protocols(frozen: dict[str, Any]) -> tuple[str, str]:
    paths = [str(item.get("path") or "") for item in frozen.get("protocols") or []]
    base = [path for path in paths if Path(path).name == "free_speech_v5_promotion_protocol.json"]
    v10 = [path for path in paths if Path(path).name == "free_speech_v10_consumer_promotion_protocol.json"]
    if len(base) != 1 or len(v10) != 1:
        raise ValueError("freeze must contain exactly the frozen v5 and v10 promotion protocols")
    return base[0], v10[0]


def run(
    freeze_json: str | Path,
    output_dir: str | Path,
    *,
    asr_model: str = "small",
    asr_provider: str = "auto",
    raters: list[str] | None = None,
    ratings_per_presentation: int = 5,
    completed_ratings_csv: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    verification = verify_freeze(freeze_json)
    if not verification.get("verification_ok"):
        raise RuntimeError("confirmatory freeze verification failed; do not run held evaluation on drifted assets")

    frozen = json.loads(Path(freeze_json).read_text(encoding="utf-8"))
    base_protocol, v10_protocol = _protocols(frozen)
    manifest = str((frozen.get("manifest") or {}).get("path") or "")
    audio_root = str(frozen.get("audio_root") or "")
    if not manifest or not audio_root:
        raise ValueError("freeze is missing manifest or audio_root")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readiness = run_v10(
        manifest,
        out_dir / "readiness_v10",
        audio_root=audio_root,
        asr_model=asr_model,
        asr_provider=asr_provider,
        raters=raters,
        ratings_per_presentation=ratings_per_presentation,
        completed_ratings_csv=completed_ratings_csv,
        base_protocol_json=base_protocol,
        v10_protocol_json=v10_protocol,
        resume=resume,
    )

    report = {
        "schema": RUN_SCHEMA,
        "freeze_json": str(freeze_json),
        "freeze_fingerprint_sha256": frozen.get("freeze_fingerprint_sha256"),
        "freeze_verification": verification,
        "readiness_v10": readiness,
        "held_set_used_for_threshold_tuning": False,
        "protocol_retuning_allowed": False,
        "product_score_changed": False,
        "score_contract_changed": False,
        "decision": readiness.get("decision"),
        "next_action": readiness.get("next_action"),
    }
    summary = out_dir / "confirmatory_run_summary_v11.json"
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["summary_json"] = str(summary)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("freeze_json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--asr-provider", default="auto")
    parser.add_argument("--raters", default="", help="comma-separated anonymized rater ids")
    parser.add_argument("--ratings-per-presentation", type=int, default=5)
    parser.add_argument("--ratings-csv", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = run(
        args.freeze_json,
        args.out_dir,
        asr_model=args.asr_model,
        asr_provider=args.asr_provider,
        raters=[item.strip() for item in args.raters.split(",") if item.strip()],
        ratings_per_presentation=args.ratings_per_presentation,
        completed_ratings_csv=args.ratings_csv,
        resume=not args.no_resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

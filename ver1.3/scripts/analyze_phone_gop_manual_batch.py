#!/usr/bin/env python3
"""Analyze a future manual GOP recording batch without hand-opening JSON files.

Expected score artifacts are `<clip_id>.json` files produced by the phone-GOP
shadow runner. Missing clips are reported and skipped; the script never asks for
replacement recordings merely because a file is absent.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.phone_gop_batch_analysis import analyze_manual_gop_batch  # noqa: E402


DEFAULT_MANIFEST = ROOT / "data" / "phone_gop_manual_validation_manifest_v2.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "phone_gop_manual_batch_analysis_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze grouped Japanese phone-GOP shadow artifacts")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--gop-dir", required=True, help="Directory containing <clip_id>.json GOP artifacts")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    gop_dir = Path(args.gop_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not gop_dir.exists():
        raise FileNotFoundError(gop_dir)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))

    payloads = {}
    for row in manifest_rows:
        clip_id = str(row.get("clip_id") or "").strip()
        if not clip_id:
            continue
        path = gop_dir / f"{clip_id}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"invalid GOP JSON for {clip_id}: {exc}") from exc
        payloads[clip_id] = payload

    report = analyze_manual_gop_batch(manifest_rows, payloads)
    report["input"] = {
        "manifest": str(manifest_path),
        "gop_dir": str(gop_dir),
        "payload_count": len(payloads),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {output}")
    print(f"groups analyzed: {report['groups_analyzed']}")
    print(f"complete normal/error groups: {report['complete_error_groups']}")
    print(f"missing clips: {report['missing_clip_count']}")
    if report["directional_error_groups_evaluated"]:
        print(
            "error-direction hits: "
            f"{report['directional_error_groups_with_lower_posterior_margin']} / "
            f"{report['directional_error_groups_evaluated']}"
        )


if __name__ == "__main__":
    main()

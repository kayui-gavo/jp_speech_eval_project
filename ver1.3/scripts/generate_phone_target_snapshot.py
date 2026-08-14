#!/usr/bin/env python3
"""Generate deterministic pilot target phones without loading any acoustic model.

This is a Stage-0 artifact. It lets us inspect/freeze the exact pyopenjtalk-plus
phone sequences before asking anyone to record the manual GOP battery.
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

from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402


DEFAULT_MANIFEST = ROOT / "data" / "phone_gop_manual_validation_manifest_v2.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "phone_target_snapshot_candidate_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pyopenjtalk-plus pilot target phone snapshot")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    rows: dict[str, dict[str, object]] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = str(row.get("target_text") or "").strip()
            if not text:
                continue
            evidence = build_japanese_target_evidence(text)
            key = text
            if key in rows:
                continue
            rows[key] = {
                "text": text,
                "kana": evidence.reading_kana,
                "phones": evidence.phones,
                "moras": evidence.moras,
                "frontend_distribution": evidence.frontend_distribution,
                "frontend_version": evidence.frontend_version,
                "frontend_ambiguous": evidence.frontend_ambiguous,
                "warnings": evidence.warnings,
            }

    distributions = sorted({str(row["frontend_distribution"]) for row in rows.values()})
    ambiguous = any(bool(row["frontend_ambiguous"]) for row in rows.values())
    warnings = sorted({str(w) for row in rows.values() for w in row["warnings"]})
    payload = {
        "schema": "japanese_phone_target_snapshot_candidate_v1",
        "frozen": False,
        "target_count": len(rows),
        "frontend_distributions": distributions,
        "frontend_ambiguous": ambiguous,
        "warnings": warnings,
        "targets": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output} ({len(rows)} unique targets)")

    if ambiguous or distributions != ["pyopenjtalk-plus"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

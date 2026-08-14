#!/usr/bin/env python3
"""Probe a locally obtained UME-JRF corpus without parsing unknown labels.

This script never downloads UME-JRF. It inventories the supplied directory,
records corpus-internal documentation when present, and emits the research-only
license guard needed before any label/audio adapter is implemented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.ume_jrf_research import (  # noqa: E402
    probe_ume_jrf_layout,
    research_guard_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a local UME-JRF research corpus layout")
    parser.add_argument("corpus_root")
    parser.add_argument("--output", default="outputs/ume_jrf_layout_probe.json")
    args = parser.parse_args()

    probe = probe_ume_jrf_layout(args.corpus_root)
    payload = {
        "schema": "ume_jrf_layout_probe_v1",
        "guard": research_guard_metadata(),
        "probe": probe.to_dict(),
        "network_download_performed": False,
        "label_parser_activated": False,
        "product_score_changed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print("available:", probe.available)
    print("grading schema:", probe.grading_schema_status)
    print("automatic label parsing allowed:", probe.automatic_label_parsing_allowed)
    print("COMMERCIAL PRODUCT USE: BLOCKED BY CORPUS LICENSE")


if __name__ == "__main__":
    main()

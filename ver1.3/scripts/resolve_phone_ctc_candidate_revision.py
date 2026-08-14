#!/usr/bin/env python3
"""Resolve immutable Hugging Face revisions for shadow phone-CTC candidates.

This is a Stage-0 reproducibility helper. It performs metadata lookup only and
writes the exact repository SHA that should be reviewed and pinned before any
candidate model is trusted in a benchmark. Nothing here affects product scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_MODELS = [
    "TylorShine/distilhubert-hiragana-ctc",
    "TylorShine/wavlm-base-plus-hiragana-ctc",
]


def resolve_models(model_ids: List[str]) -> Dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    rows = []
    for model_id in model_ids:
        info = api.model_info(model_id, files_metadata=False)
        rows.append(
            {
                "model_id": model_id,
                "resolved_revision": str(info.sha or ""),
                "last_modified": str(info.last_modified or ""),
                "private": bool(info.private),
                "gated": bool(info.gated),
                "library_name": info.library_name,
                "pipeline_tag": info.pipeline_tag,
                "tags": sorted(str(tag) for tag in (info.tags or [])),
                "purpose": "shadow_phone_ctc_candidate_revision_discovery",
            }
        )
    return {
        "schema": "phone_ctc_candidate_revision_v1",
        "score_mapped": False,
        "product_calibrated": False,
        "models": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/phone_ctc_candidate_revisions.json")
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()

    model_ids = args.models or list(DEFAULT_MODELS)
    payload = resolve_models(model_ids)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in payload["models"]:
        print(f"{row['model_id']} -> {row['resolved_revision']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def default_verified_targets_path() -> Path:
    env_path = os.environ.get("JP_SPEECH_EVAL_VERIFIED_TARGETS")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "configs" / "verified_accent_targets.json"


def load_verified_targets(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    target_path = Path(path) if path is not None else default_verified_targets_path()
    if not target_path.exists():
        return {}
    if target_path.stat().st_size == 0:
        return {}
    with target_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Verified accent target file must be a JSON object: {target_path}")
    return {
        str(text): dict(entry)
        for text, entry in raw.items()
        if isinstance(entry, dict)
    }


def lookup_verified_target(text: str, path: str | Path | None = None) -> Optional[Dict[str, Any]]:
    return load_verified_targets(path).get(text)

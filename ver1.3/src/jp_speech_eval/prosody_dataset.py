from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ProsodyTrial:
    """One ABX trial over three audio files."""

    trial_id: str
    a_audio: str
    b_audio: str
    x_audio: str
    x_label: str
    context_mode: str
    reference_source: str = "native"
    regions: Dict[str, List[float]] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class ProsodyPair:
    """Metadata and optional trials for one prosodic minimal pair."""

    pair_id: str
    contrast_type: str
    word_a: str
    accent_a: str
    word_b: str
    accent_b: str
    carrier_a: str
    carrier_b: str
    target_region: str
    notes: str = ""
    trials: List[ProsodyTrial] = field(default_factory=list)


@dataclass(frozen=True)
class ProsodyDataset:
    """Validated minimal-pair dataset used by Prosodic ABX diagnostics."""

    pairs: List[ProsodyPair]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _trial_from_dict(pair_id: str, index: int, raw: Dict[str, Any]) -> ProsodyTrial:
    required = ("a_audio", "b_audio", "x_audio", "x_label", "context_mode")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"{pair_id}.trials[{index}] missing fields: {missing}")
    x_label = str(raw["x_label"]).upper()
    if x_label not in {"A", "B"}:
        raise ValueError(f"{pair_id}.trials[{index}].x_label must be 'A' or 'B'")
    return ProsodyTrial(
        trial_id=str(raw.get("trial_id") or f"{pair_id}_{index:03d}"),
        a_audio=str(raw["a_audio"]),
        b_audio=str(raw["b_audio"]),
        x_audio=str(raw["x_audio"]),
        x_label=x_label,
        context_mode=str(raw["context_mode"]),
        reference_source=str(raw.get("reference_source") or "native"),
        regions=dict(raw.get("regions") or {}),
        notes=str(raw.get("notes") or ""),
    )


def _pair_from_dict(raw: Dict[str, Any]) -> ProsodyPair:
    required = (
        "pair_id",
        "contrast_type",
        "word_a",
        "accent_a",
        "word_b",
        "accent_b",
        "carrier_a",
        "carrier_b",
        "target_region",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"prosody pair missing fields: {missing}")
    pair_id = str(raw["pair_id"])
    trials = [_trial_from_dict(pair_id, i, item) for i, item in enumerate(raw.get("trials") or [])]
    return ProsodyPair(
        pair_id=pair_id,
        contrast_type=str(raw["contrast_type"]),
        word_a=str(raw["word_a"]),
        accent_a=str(raw["accent_a"]),
        word_b=str(raw["word_b"]),
        accent_b=str(raw["accent_b"]),
        carrier_a=str(raw["carrier_a"]),
        carrier_b=str(raw["carrier_b"]),
        target_region=str(raw["target_region"]),
        notes=str(raw.get("notes") or ""),
        trials=trials,
    )


def load_prosody_dataset(path: str | Path) -> ProsodyDataset:
    """Load a list-style or object-style prosody dataset JSON file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        pairs_raw = raw
        metadata: Dict[str, Any] = {}
    elif isinstance(raw, dict):
        pairs_raw = raw.get("pairs") or []
        metadata = dict(raw.get("metadata") or {})
    else:
        raise ValueError("prosody dataset must be a JSON list or object")

    pairs = [_pair_from_dict(item) for item in pairs_raw]
    if not pairs:
        raise ValueError(f"prosody dataset has no pairs: {path}")
    return ProsodyDataset(pairs=pairs, metadata=metadata)


def iter_trials(dataset: ProsodyDataset, mode: Optional[str] = None) -> Iterable[tuple[ProsodyPair, ProsodyTrial]]:
    """Yield explicit trials, optionally filtered by `context_mode`."""

    for pair in dataset.pairs:
        for trial in pair.trials:
            if mode is None or trial.context_mode == mode:
                yield pair, trial


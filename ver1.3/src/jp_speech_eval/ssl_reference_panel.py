"""Explicit multi-native reference panels for SSL research shadows.

The panel is intentionally a research/provenance object, not a ProductScore
asset. It lets a fixed target use several human/native references without
hard-coding corpus paths into the evaluator. A panel can live outside the repo;
relative audio paths resolve from the panel JSON location.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


SSL_REFERENCE_PANEL_SCHEMA = "ssl_reference_panel_v1"
ALLOWED_REFERENCE_KINDS = {"human_native", "human_near_native", "tts_fallback"}


@dataclass(frozen=True)
class SSLReference:
    reference_id: str
    target_text: str
    audio_path: str
    speaker_id: str = ""
    reference_kind: str = "human_native"
    provenance: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "reference_id": self.reference_id,
            "target_text": self.target_text,
            "audio_path": self.audio_path,
            "speaker_id": self.speaker_id,
            "reference_kind": self.reference_kind,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class SSLReferencePanel:
    panel_id: str
    schema: str
    references: tuple[SSLReference, ...]
    source_path: str = ""

    def references_for_target(
        self,
        target_text: str,
        *,
        allow_tts_fallback: bool = False,
    ) -> List[SSLReference]:
        target = str(target_text or "").strip()
        selected = [item for item in self.references if item.target_text.strip() == target]
        if not allow_tts_fallback:
            human = [item for item in selected if item.reference_kind != "tts_fallback"]
            if human:
                selected = human
            else:
                selected = []
        return selected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "panel_id": self.panel_id,
            "source_path": self.source_path,
            "reference_count": len(self.references),
        }


def _canonical_panel_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _reference_from_mapping(
    row: Mapping[str, Any],
    *,
    base_dir: Path,
) -> SSLReference:
    reference_id = str(row.get("reference_id") or "").strip()
    target_text = str(row.get("target_text") or "").strip()
    raw_path = str(row.get("audio_path") or "").strip()
    speaker_id = str(row.get("speaker_id") or "").strip()
    reference_kind = str(row.get("reference_kind") or "human_native").strip().lower()
    provenance = str(row.get("provenance") or "").strip()
    if not reference_id:
        raise ValueError("reference_id is required")
    if not target_text:
        raise ValueError(f"target_text is required for reference {reference_id}")
    if not raw_path:
        raise ValueError(f"audio_path is required for reference {reference_id}")
    if reference_kind not in ALLOWED_REFERENCE_KINDS:
        raise ValueError(
            f"unsupported reference_kind {reference_kind!r} for {reference_id}; "
            f"allowed={sorted(ALLOWED_REFERENCE_KINDS)}"
        )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return SSLReference(
        reference_id=reference_id,
        target_text=target_text,
        audio_path=str(path),
        speaker_id=speaker_id,
        reference_kind=reference_kind,
        provenance=provenance,
    )


def load_ssl_reference_panel(
    path: str | Path,
    *,
    require_audio_exists: bool = True,
) -> SSLReferencePanel:
    panel_path = Path(path).expanduser().resolve()
    payload = json.loads(panel_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("SSL reference panel JSON must be an object")
    schema = str(payload.get("schema") or SSL_REFERENCE_PANEL_SCHEMA)
    if schema != SSL_REFERENCE_PANEL_SCHEMA:
        raise ValueError(f"unsupported SSL reference panel schema: {schema}")
    raw_references = payload.get("references")
    if not isinstance(raw_references, list) or not raw_references:
        raise ValueError("SSL reference panel requires a non-empty references list")
    references: List[SSLReference] = []
    ids: set[str] = set()
    for raw in raw_references:
        if not isinstance(raw, Mapping):
            raise ValueError("every SSL reference entry must be an object")
        item = _reference_from_mapping(raw, base_dir=panel_path.parent)
        if item.reference_id in ids:
            raise ValueError(f"duplicate SSL reference_id: {item.reference_id}")
        ids.add(item.reference_id)
        if require_audio_exists and not Path(item.audio_path).is_file():
            raise FileNotFoundError(f"reference audio not found: {item.audio_path}")
        references.append(item)

    identity_payload = {
        "schema": schema,
        "references": [
            {
                "reference_id": item.reference_id,
                "target_text": item.target_text,
                "speaker_id": item.speaker_id,
                "reference_kind": item.reference_kind,
                "provenance": item.provenance,
                # Keep identity path-independent so moving the same panel does
                # not create a fake new reference generation.
                "audio_name": Path(item.audio_path).name,
            }
            for item in references
        ],
    }
    panel_id = str(payload.get("panel_id") or "").strip() or (
        "sslref_" + _canonical_panel_hash(identity_payload)
    )
    return SSLReferencePanel(
        panel_id=panel_id,
        schema=schema,
        references=tuple(references),
        source_path=str(panel_path),
    )


def reference_panel_template(target_text: str = "") -> Dict[str, Any]:
    return {
        "schema": SSL_REFERENCE_PANEL_SCHEMA,
        "panel_id": "",
        "references": [
            {
                "reference_id": "native_01",
                "target_text": target_text,
                "audio_path": "references/native_01.wav",
                "speaker_id": "native_01",
                "reference_kind": "human_native",
                "provenance": "consented_reference_recording",
            }
        ],
    }

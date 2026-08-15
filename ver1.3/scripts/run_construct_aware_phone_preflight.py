#!/usr/bin/env python3
"""Run role-safe Japanese phone research features on bundled audio.

This preflight exists to verify the *construct separation* rather than to score
a learner. Ordinary phones may feed a future clarity criterion model, while
``N`` / ``cl`` are routed to special-mora timing support and must not silently
enter ordinary clarity features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.audio_features import load_audio  # noqa: E402
from jp_speech_eval.japanese_phoneme_gop import (  # noqa: E402
    JapanesePhoneCtcBackend,
    project_japanese_ctc_logits,
    sanitize_canonical_phones,
)
from jp_speech_eval.japanese_segmentation_free_norm import compute_japanese_construct_aware_norm_features  # noqa: E402
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.phone_construct_roles import build_construct_role_view  # noqa: E402
from jp_speech_eval.phone_criterion_features import build_phone_criterion_feature_bundle  # noqa: E402
from jp_speech_eval.segmentation_free_gop import evaluate_backend_fgop_sf_sd_shadow  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


WAV = ROOT / "assets" / "reference_cache" / "ramen_kudasai_aivis.ref.wav"
TEXT = "ラーメンをください。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/construct_aware_phone_preflight.json")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _logical_logits(backend: JapanesePhoneCtcBackend, speech: np.ndarray):
    backend._load()
    inputs = backend.processor(speech, sampling_rate=16000, return_tensors="pt")
    model_inputs = {key: value.to(backend.device) for key, value in inputs.items()}
    with backend._torch.no_grad():
        output = backend.model(**model_inputs)
    raw = output.logits.squeeze(0).detach().cpu().numpy()
    logical, vocab, provenance = project_japanese_ctc_logits(raw, backend.vocabulary())
    return logical, vocab, int(vocab["PAD"]), provenance


def main() -> None:
    args = parse_args()
    audio = load_audio(str(WAV), sr=16000)
    speech, region = trim_to_speech(audio.y, audio.sr)
    target = build_japanese_target_evidence(TEXT)
    phones, dropped = sanitize_canonical_phones(target.phones)
    backend = JapanesePhoneCtcBackend(
        device=args.device,
        local_files_only=not bool(args.allow_download),
    )

    enumerated = evaluate_backend_fgop_sf_sd_shadow(backend, speech, phones, sr=16000)
    logical, vocab, blank_id, provenance = _logical_logits(
        backend, np.asarray(speech, dtype=np.float32)
    )
    normalized = compute_japanese_construct_aware_norm_features(
        logical,
        phones,
        vocab=vocab,
        blank_id=blank_id,
        model_id=str(backend.model_id),
        revision=str(backend.revision),
    )
    criterion = build_phone_criterion_feature_bundle(enumerated, normalized)
    role_view = build_construct_role_view(criterion)

    ordinary = [row for row in role_view.get("rows", []) if row.get("clarity_primary_eligible")]
    special = [row for row in role_view.get("rows", []) if row.get("construct_role") == "special_mora_timing_support"]
    payload = {
        "schema": "construct_aware_phone_preflight_v1",
        "audio": str(WAV.relative_to(ROOT)),
        "text": TEXT,
        "speech_region": region.to_dict(),
        "phones": phones,
        "dropped_nonsegmental_tokens": dropped,
        "logical_allophone_groups": {key: value for key, value in provenance.items() if len(value) > 1},
        "enumerated": enumerated.to_dict(),
        "construct_aware_normalized": normalized.to_dict(),
        "criterion_bundle": criterion.to_dict(),
        "construct_role_view": role_view,
        "summary": {
            "ordinary_clarity_phone_count": len(ordinary),
            "special_mora_support_count": len(special),
            "special_mora_support_phones": [row["canonical_phone"] for row in special],
            "special_mora_has_ordinary_clarity_primary_feature": any(row.get("clarity_primary_eligible") for row in special),
            "score_mapped": False,
            "product_calibrated": False,
            "product_score_changed": False,
            "human_recording_gate_changed": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    if not enumerated.available or not normalized.available or not criterion.available or not role_view.get("available"):
        raise SystemExit("construct-aware feature family unavailable")
    if payload["summary"]["special_mora_has_ordinary_clarity_primary_feature"]:
        raise SystemExit("special mora leaked into ordinary clarity primary features")
    print("PRODUCT SCORE: UNCHANGED / RESEARCH ONLY")
    print("HUMAN RECORDING GATE: UNCHANGED / BLOCKED")


if __name__ == "__main__":
    main()

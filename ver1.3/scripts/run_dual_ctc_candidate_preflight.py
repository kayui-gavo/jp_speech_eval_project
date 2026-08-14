#!/usr/bin/env python3
"""Run a pinned alternative Japanese dual-CTC model on bundled audio.

This is a Stage-0 shadow comparison against the existing Beatrice preflight.
It uses only repository-bundled audio and never changes a product score.  The
artifact includes both the transparent enumerated LPP/LPR feature family and
the paper-aligned SD alternative-graph normalized forward/Occ(i) diagnostics.
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
from jp_speech_eval.dual_ctc_phone_candidate import (  # noqa: E402
    DISTILHUBERT_DUAL_CTC_MODEL,
    DualCtcPhoneCandidateBackend,
)
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.segmentation_free_gop_norm import compute_segmentation_free_norm_features  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


BUNDLED_AUDIO = ROOT / "assets" / "reference_cache" / "ramen_kudasai_aivis.ref.wav"
CORRECT_TEXT = "ラーメンをください。"
WRONG_TEXT = "コーヒーをください。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DISTILHUBERT_DUAL_CTC_MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="outputs/dual_ctc_candidate_preflight.json")
    return parser.parse_args()


def _evaluate(backend: DualCtcPhoneCandidateBackend, speech: np.ndarray, text: str) -> dict:
    target = build_japanese_target_evidence(text)
    frame = backend.evaluate_frame_local(speech, target.phones, sr=16000)
    sf = backend.evaluate_segmentation_free(speech, target.phones, sr=16000)

    norm_payload: dict
    try:
        logical_logits, logical_vocab, blank_id, _provenance = backend.infer_logical_phone_logits(
            speech, sr=16000
        )
        norm = compute_segmentation_free_norm_features(
            logical_logits,
            target.phones,
            vocab=logical_vocab,
            blank_id=blank_id,
            model_id=backend.model_id,
            revision=backend.revision,
        )
        norm_payload = norm.to_dict()
    except Exception as exc:
        norm_payload = {
            "available": False,
            "model_id": backend.model_id,
            "revision": backend.revision,
            "method": "paper_sd_norm_forward_v1",
            "canonical_phones": list(target.phones),
            "evidence": [],
            "summary": {
                "reason": f"norm_feature_extraction_failed:{type(exc).__name__}",
                "detail": str(exc),
            },
            "warnings": ["norm_feature_extraction_failed"],
            "score_mapped": False,
            "product_calibrated": False,
        }

    return {
        "text": text,
        "phones": target.phones,
        "frame_local": frame.to_dict(),
        "segmentation_free": sf.to_dict(),
        "segmentation_free_norm": norm_payload,
    }


def main() -> None:
    args = parse_args()
    if not BUNDLED_AUDIO.exists():
        raise SystemExit(f"bundled audio missing: {BUNDLED_AUDIO}")

    audio = load_audio(str(BUNDLED_AUDIO), sr=16000)
    speech, region = trim_to_speech(audio.y, audio.sr)
    backend = DualCtcPhoneCandidateBackend(
        model_id=args.model,
        revision=args.revision,
        device=args.device,
        local_files_only=not args.allow_download,
    )

    correct = _evaluate(backend, speech, CORRECT_TEXT)
    wrong = _evaluate(backend, speech, WRONG_TEXT)
    gain_rows = []
    for gain in (0.8, 1.2):
        result = _evaluate(backend, np.asarray(speech * gain, dtype=np.float32), CORRECT_TEXT)
        gain_rows.append({"gain": gain, "result": result})

    correct_lp = correct["segmentation_free"]["summary"].get("canonical_ctc_log_posterior")
    wrong_lp = wrong["segmentation_free"]["summary"].get("canonical_ctc_log_posterior")
    payload = {
        "schema": "dual_ctc_candidate_preflight_v2",
        "model_id": args.model,
        "revision": args.revision,
        "audio": str(BUNDLED_AUDIO.relative_to(ROOT)),
        "speech_region": region.to_dict(),
        "score_mapped": False,
        "product_calibrated": False,
        "product_score_changed": False,
        "human_recordings_used": False,
        "individual_lpr_sign_is_pronunciation_error_rule": False,
        "occ_i_is_physical_phone_duration": False,
        "correct_target": correct,
        "wrong_target": wrong,
        "gain_controls": gain_rows,
        "sequence_logposterior_gap_correct_minus_wrong": (
            float(correct_lp) - float(wrong_lp)
            if correct_lp is not None and wrong_lp is not None
            else None
        ),
        "interpretation": "engineering_shadow_features_require_labeled_downstream_validation",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"model: {args.model}@{args.revision}")
    print(f"correct-minus-wrong sequence log posterior: {payload['sequence_logposterior_gap_correct_minus_wrong']}")
    norm_summary = correct["segmentation_free_norm"].get("summary", {})
    print("correct Occ(i) range:", norm_summary.get("occ_i_min"), norm_summary.get("occ_i_max"))
    print("PRODUCT SCORE: UNCHANGED / SHADOW ONLY")


if __name__ == "__main__":
    main()

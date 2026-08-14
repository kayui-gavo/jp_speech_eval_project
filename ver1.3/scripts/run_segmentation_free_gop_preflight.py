#!/usr/bin/env python3
"""Evaluate alignment-free Japanese phone features on bundled audio only.

The artifact deliberately does not interpret the sign of an individual LPR as
a pronunciation-correctness decision. Published FGOP-SF work uses the joint
LPP/LPR feature vector in downstream pronunciation assessment; this script is
therefore an engineering/feature preflight, not a clean-phone threshold test.

Alongside the transparent enumerated LPP/LPR extractor, this preflight runs an
independent NumPy reimplementation of the published normalized SD
alternative-graph forward recursion to expose ``Occ(i)``. It also joins both
families into a strict criterion-ready feature bundle for later labeled work.
``Occ(i)`` is graph occupancy/activation, never a physical phone duration.
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
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.phone_criterion_features import (  # noqa: E402
    build_phone_criterion_feature_bundle,
    compare_shared_suffix_locality,
)
from jp_speech_eval.segmentation_free_gop import evaluate_backend_fgop_sf_sd_shadow  # noqa: E402
from jp_speech_eval.segmentation_free_gop_norm import compute_segmentation_free_norm_features  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


DEFAULT_WAV = ROOT / "assets" / "reference_cache" / "ramen_kudasai_aivis.ref.wav"
DEFAULT_OUTPUT = ROOT / "outputs" / "segmentation_free_gop_preflight_v1.json"
CORRECT_TEXT = "ラーメンをください。"
WRONG_TEXT = "コーヒーをください。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alignment-free Japanese phone feature preflight")
    parser.add_argument("--wav", default=str(DEFAULT_WAV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--wrong-text", default=WRONG_TEXT)
    return parser.parse_args()


def _weakest_rows(result, limit: int = 8):
    if not result.available:
        return []
    rows = sorted(
        result.evidence,
        key=lambda row: row.best_noncanonical_log_posterior_ratio,
    )
    return [
        {
            "phone_index": row.phone_index,
            "phone": row.canonical_phone,
            "best_noncanonical_type": row.best_noncanonical_alternative_type,
            "best_noncanonical_phone": row.best_noncanonical_alternative_phone,
            "best_noncanonical_lpr": row.best_noncanonical_log_posterior_ratio,
            "deletion_lpr": row.deletion_log_posterior_ratio,
            "gop_sf_sd": row.gop_sf_sd,
        }
        for row in rows[:limit]
    ]


def _norm_features(
    backend: JapanesePhoneCtcBackend,
    speech: np.ndarray,
    canonical_phones,
    *,
    sr: int,
):
    backend._load()  # package-private research backend; pinned contract validated
    waveform = np.asarray(speech, dtype=np.float32).reshape(-1)
    clean_phones, _dropped = sanitize_canonical_phones(canonical_phones)
    inputs = backend.processor(waveform, sampling_rate=sr, return_tensors="pt")
    model_inputs = {key: value.to(backend.device) for key, value in inputs.items()}
    with backend._torch.no_grad():
        output = backend.model(**model_inputs)
    raw_logits = output.logits.squeeze(0).detach().cpu().numpy()
    logical_logits, logical_vocab, _projection = project_japanese_ctc_logits(
        raw_logits, backend.vocabulary()
    )
    if "PAD" not in logical_vocab:
        raise RuntimeError("logical blank token missing from Beatrice projection")
    return compute_segmentation_free_norm_features(
        logical_logits,
        clean_phones,
        vocab=logical_vocab,
        blank_id=int(logical_vocab["PAD"]),
        model_id=str(backend.model_id),
        revision=str(backend.revision),
    )


def _failed_bundle(reason: str, model_id: str, revision: str) -> dict:
    return {
        "available": False,
        "schema": "phone_criterion_feature_bundle_v1",
        "model_id": model_id,
        "revision": revision,
        "canonical_phones": [],
        "substitution_phone_inventory": [],
        "rows": [],
        "summary": {"reason": reason, "product_score_changed": False},
        "warnings": [reason],
        "score_mapped": False,
        "product_calibrated": False,
    }


def main() -> None:
    args = parse_args()
    audio = load_audio(str(Path(args.wav)), sr=16000)
    speech, region = trim_to_speech(audio.y, audio.sr)
    correct_target = build_japanese_target_evidence(CORRECT_TEXT)
    wrong_target = build_japanese_target_evidence(str(args.wrong_text))
    backend = JapanesePhoneCtcBackend(
        device=args.device,
        local_files_only=not args.allow_download,
    )

    correct = evaluate_backend_fgop_sf_sd_shadow(
        backend, speech, correct_target.phones, sr=audio.sr
    )
    wrong = evaluate_backend_fgop_sf_sd_shadow(
        backend, speech, wrong_target.phones, sr=audio.sr
    )
    model_id = str(backend.model_id)
    revision = str(backend.revision)
    try:
        correct_norm_obj = _norm_features(
            backend, speech, correct_target.phones, sr=audio.sr
        )
        wrong_norm_obj = _norm_features(
            backend, speech, wrong_target.phones, sr=audio.sr
        )
        correct_bundle_obj = build_phone_criterion_feature_bundle(correct, correct_norm_obj)
        wrong_bundle_obj = build_phone_criterion_feature_bundle(wrong, wrong_norm_obj)
        correct_norm = correct_norm_obj.to_dict()
        wrong_norm = wrong_norm_obj.to_dict()
        correct_bundle = correct_bundle_obj.to_dict()
        wrong_bundle = wrong_bundle_obj.to_dict()
        shared_suffix = compare_shared_suffix_locality(
            correct_bundle_obj, wrong_bundle_obj
        )
    except Exception as exc:
        reason = f"norm_or_bundle_extraction_failed:{type(exc).__name__}"
        failed = {
            "available": False,
            "model_id": model_id,
            "revision": revision,
            "method": "paper_sd_norm_forward_v1",
            "evidence": [],
            "summary": {"reason": reason, "detail": str(exc)},
            "warnings": [reason],
            "score_mapped": False,
            "product_calibrated": False,
        }
        correct_norm = dict(failed)
        wrong_norm = dict(failed)
        correct_bundle = _failed_bundle(reason, model_id, revision)
        wrong_bundle = _failed_bundle(reason, model_id, revision)
        shared_suffix = {"available": False, "reason": reason}

    payload = {
        "schema": "segmentation_free_gop_bundled_preflight_v3",
        "product_score_changed": False,
        "score_mapped": False,
        "human_recording_allowed": False,
        "individual_lpr_sign_is_pronunciation_error_rule": False,
        "occ_i_is_physical_phone_duration": False,
        "cross_model_raw_feature_averaging_allowed": False,
        "speech_region": region.to_dict(),
        "correct_target": correct_target.to_dict(),
        "wrong_target": wrong_target.to_dict(),
        "correct": correct.to_dict(),
        "wrong": wrong.to_dict(),
        "correct_norm": correct_norm,
        "wrong_norm": wrong_norm,
        "correct_criterion_feature_bundle": correct_bundle,
        "wrong_criterion_feature_bundle": wrong_bundle,
        "shared_suffix_locality": shared_suffix,
        "diagnostic": {
            "correct_weakest_rows": _weakest_rows(correct),
            "wrong_weakest_rows": _weakest_rows(wrong),
            "forced_viterbi_not_required_for_features": True,
            "individual_lpr_sign_is_pronunciation_error_rule": False,
            "noncanonical_win_count_is_stage0_failure_gate": False,
            "downstream_labeled_interpretation_required": True,
            "note": (
                "This artifact evaluates enumerated LPP/LPR substitution+deletion features, "
                "paper-aligned SD graph/Occ(i), and their strict criterion-ready join. An "
                "individual feature is not a direct mispronunciation label. Nothing is mapped "
                "to /100. Shared-suffix comparison is implementation locality evidence only."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print("correct available:", correct.available)
    if correct.available:
        print(
            "diagnostic positions with a higher-posterior noncanonical SD alternative:",
            correct.summary["phones_where_noncanonical_outscores_canonical"],
            "(feature diagnostic only; not a pronunciation-error count)",
        )
        for row in _weakest_rows(correct, limit=5):
            print(row)
    print(
        "correct Occ(i) range:",
        correct_norm.get("summary", {}).get("occ_i_min"),
        correct_norm.get("summary", {}).get("occ_i_max"),
    )
    print("criterion bundle available:", correct_bundle.get("available"))
    print("shared suffix locality:", shared_suffix)
    print("HUMAN RECORDING GATE: BLOCKED (awaits labeled criterion / Stage-0 promotion)")
    if not correct.available or not wrong.available:
        raise SystemExit(2)
    if not bool(correct_norm.get("available")) or not bool(wrong_norm.get("available")):
        raise SystemExit(3)
    if not bool(correct_bundle.get("available")) or not bool(wrong_bundle.get("available")):
        raise SystemExit(4)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a pinned alternative Japanese dual-CTC model on bundled audio.

This is a Stage-0 shadow comparison against the existing Beatrice preflight.
It uses only repository-bundled audio and never changes a product score. The
artifact includes frame-local logit/posterior evidence, alignment-free phone
features, Japanese position-masked SD/Occ(i), a strict hybrid criterion bundle,
and explicit CTC posterior peakiness/uncertainty diagnostics.

Before criterion assembly, canonical kana is aligned to the actual canonical
phone sequence to infer construct roles. This prevents a repeated vowel that
realizes a lexical long-vowel extension from being silently treated as ordinary
segmental clarity. If construct-role inference cannot be established exactly,
the criterion/hybrid branch fails closed while independent model diagnostics
remain available.

Posterior peakiness uses the *acoustic* phone inventory, including special
morae N/cl. That inventory is intentionally broader than the ordinary clarity
competitor set.
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
from jp_speech_eval.ctc_posterior_diagnostics import compute_ctc_posterior_diagnostics  # noqa: E402
from jp_speech_eval.dual_ctc_phone_candidate import (  # noqa: E402
    DISTILHUBERT_DUAL_CTC_MODEL,
    DualCtcPhoneCandidateBackend,
)
from jp_speech_eval.hybrid_phone_criterion_features import (  # noqa: E402
    SCHEMA as HYBRID_SCHEMA,
    build_hybrid_phone_criterion_bundle,
)
from jp_speech_eval.japanese_phone_inventory import (  # noqa: E402
    acoustic_phone_token_ids,
    inventory_semantics,
)
from jp_speech_eval.japanese_phone_roles import infer_phone_construct_roles  # noqa: E402
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.phone_criterion_features import (  # noqa: E402
    SCHEMA as CRITERION_SCHEMA,
    build_phone_criterion_feature_bundle,
)
from jp_speech_eval.segmentation_free_gop_norm import (  # noqa: E402
    METHOD as NORM_METHOD,
    compute_segmentation_free_norm_features,
)
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


def _unavailable_bundle(reason: str, backend: DualCtcPhoneCandidateBackend) -> dict:
    return {
        "available": False,
        "schema": CRITERION_SCHEMA,
        "model_id": backend.model_id,
        "revision": backend.revision,
        "canonical_phones": [],
        "substitution_phone_inventory": [],
        "rows": [],
        "summary": {"reason": reason, "product_score_changed": False},
        "warnings": [reason],
        "score_mapped": False,
        "product_calibrated": False,
    }


def _unavailable_hybrid(reason: str, backend: DualCtcPhoneCandidateBackend) -> dict:
    return {
        "available": False,
        "schema": HYBRID_SCHEMA,
        "model_id": backend.model_id,
        "revision": backend.revision,
        "canonical_phones": [],
        "rows": [],
        "summary": {"reason": reason, "product_score_changed": False},
        "warnings": [reason],
        "score_mapped": False,
        "product_calibrated": False,
    }


def _unavailable_diagnostics(reason: str) -> dict:
    return {
        "available": False,
        "schema": "ctc_posterior_diagnostics_v1",
        "summary": {
            "reason": reason,
            "interpretation": "model_posterior_peakiness_and_uncertainty_not_pronunciation_quality",
            "product_score_changed": False,
        },
        "warnings": [reason],
        "score_mapped": False,
        "product_calibrated": False,
    }


def _evaluate(backend: DualCtcPhoneCandidateBackend, speech: np.ndarray, text: str) -> dict:
    target = build_japanese_target_evidence(text)
    frame = backend.evaluate_frame_local(speech, target.phones, sr=16000)
    sf = backend.evaluate_segmentation_free(speech, target.phones, sr=16000)
    role_result = infer_phone_construct_roles(target.reading_kana, sf.canonical_phones)

    norm_payload: dict
    criterion_payload: dict
    hybrid_payload: dict
    posterior_payload: dict
    try:
        logical_logits, logical_vocab, blank_id, _provenance = backend.infer_logical_phone_logits(
            speech, sr=16000
        )
        norm = compute_segmentation_free_norm_features(
            logical_logits,
            sf.canonical_phones,
            vocab=logical_vocab,
            blank_id=blank_id,
            model_id=backend.model_id,
            revision=backend.revision,
        )
        norm_payload = norm.to_dict()

        if role_result.available:
            criterion = build_phone_criterion_feature_bundle(
                sf,
                norm,
                construct_roles=role_result.roles,
            )
            hybrid = build_hybrid_phone_criterion_bundle(frame, criterion)
            criterion_payload = criterion.to_dict()
            hybrid_payload = hybrid.to_dict()
        else:
            role_reason = str(role_result.summary.get("reason") or "unknown")
            reason = f"target_construct_role_inference_failed:{role_reason}"
            criterion_payload = _unavailable_bundle(reason, backend)
            hybrid_payload = _unavailable_hybrid(reason, backend)

        phone_ids = acoustic_phone_token_ids(logical_vocab, blank_id=blank_id)
        posterior_payload = compute_ctc_posterior_diagnostics(
            logical_logits,
            blank_id=blank_id,
            phone_token_ids=phone_ids,
        ).to_dict()
        posterior_payload["phone_inventory_semantics"] = inventory_semantics(
            logical_vocab,
            blank_id=blank_id,
        )
    except Exception as exc:
        reason = f"norm_or_diagnostic_extraction_failed:{type(exc).__name__}"
        norm_payload = {
            "available": False,
            "model_id": backend.model_id,
            "revision": backend.revision,
            "method": NORM_METHOD,
            "canonical_phones": list(sf.canonical_phones),
            "evidence": [],
            "summary": {"reason": reason, "detail": str(exc)},
            "warnings": ["norm_or_diagnostic_extraction_failed"],
            "score_mapped": False,
            "product_calibrated": False,
        }
        criterion_payload = _unavailable_bundle(reason, backend)
        hybrid_payload = _unavailable_hybrid(reason, backend)
        posterior_payload = _unavailable_diagnostics(reason)

    return {
        "text": text,
        "phones": list(sf.canonical_phones),
        "target_evidence": target.to_dict(),
        "target_construct_roles": role_result.to_dict(),
        "frame_local": frame.to_dict(),
        "segmentation_free": sf.to_dict(),
        "segmentation_free_norm": norm_payload,
        "criterion_feature_bundle": criterion_payload,
        "hybrid_criterion_feature_bundle": hybrid_payload,
        "ctc_posterior_diagnostics": posterior_payload,
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
        "schema": "dual_ctc_candidate_preflight_v9",
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
        "ctc_peakiness_is_pronunciation_score": False,
        "ctc_peakiness_phone_inventory_includes_special_morae": True,
        "construct_roles_inferred_before_criterion_assembly": True,
        "long_vowel_extension_is_ordinary_segmental_clarity": False,
        "cross_model_raw_feature_averaging_allowed": False,
        "normalized_sd_method": NORM_METHOD,
        "criterion_schema": CRITERION_SCHEMA,
        "hybrid_criterion_schema": HYBRID_SCHEMA,
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
    print(f"normalized SD method: {NORM_METHOD}")
    print(f"criterion schema: {CRITERION_SCHEMA}")
    print(f"hybrid criterion schema: {HYBRID_SCHEMA}")
    print(f"correct-minus-wrong sequence log posterior: {payload['sequence_logposterior_gap_correct_minus_wrong']}")
    norm_summary = correct["segmentation_free_norm"].get("summary", {})
    posterior = correct["ctc_posterior_diagnostics"]
    roles = correct["target_construct_roles"]
    print("correct construct roles available:", roles.get("available"))
    print("correct long-vowel timing phones:", roles.get("summary", {}).get("long_vowel_timing_phone_count"))
    print("correct Occ(i) range:", norm_summary.get("occ_i_min"), norm_summary.get("occ_i_max"))
    print("criterion bundle available:", correct["criterion_feature_bundle"].get("available"))
    print("hybrid criterion bundle available:", correct["hybrid_criterion_feature_bundle"].get("available"))
    print("CTC top1 posterior mean:", posterior.get("top1_posterior_mean"))
    print("CTC blank-top1 fraction:", posterior.get("blank_top1_fraction"))
    print("CTC posterior acoustic phone count:", posterior.get("phone_token_count"))
    print("PRODUCT SCORE: UNCHANGED / SHADOW ONLY")


if __name__ == "__main__":
    main()

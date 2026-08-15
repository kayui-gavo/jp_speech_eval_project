#!/usr/bin/env python3
"""Automatic local phone-edit sanity test using ephemeral OpenJTalk TTS.

This test is intentionally *not* learner validity. It creates deterministic
synthetic pairs whose canonical phone strings differ by one substitution or one
deletion, then asks whether target-conditioned restricted CTC evidence moves in
the expected local direction. Generated audio is never committed or uploaded.

Each target edit is also checked against target-side construct-role metadata.
This prevents an ordinary segment substitution control, a special-mora deletion
and a long-vowel extension deletion from being silently pooled as the same
construct merely because all are represented by phone tokens.

A repeated-vowel long vowel such as /a a/ has an important sequence-level
ambiguity: deleting either adjacent /a/ yields the same shorter phone string.
The acoustic phone sequence alone therefore cannot identify which target event
was removed. For controlled cases only, the reviewed/derived construct role is
used to disambiguate the edit index. This is target provenance, not an acoustic
claim about learner correctness.

Purpose: catch implementation/search-space/construct-provenance bugs before
spending human recording time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import librosa
import numpy as np
import pyopenjtalk

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from jp_speech_eval.japanese_phone_roles import (  # noqa: E402
    LONG_VOWEL_ROLE,
    ORDINARY_ROLE,
    SPECIAL_MORA_ROLE,
    infer_phone_construct_roles,
)
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.japanese_phoneme_gop import sanitize_canonical_phones  # noqa: E402
from jp_speech_eval.restricted_segmentation_free_gop import compute_restricted_fgop_sf_sd_features  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402
from run_official_jvs_phone_ctc_anchor_preflight import BeatriceInfer  # noqa: E402


CASES = (
    {"case_id": "voicing_b_p", "target": "バスです。", "spoken_error": "パスです。", "edit_type": "substitution", "expected_construct_role": ORDINARY_ROLE},
    {"case_id": "voicing_g_k", "target": "かぎです。", "spoken_error": "かきです。", "edit_type": "substitution", "expected_construct_role": ORDINARY_ROLE},
    {"case_id": "affricate_ts_s", "target": "つきです。", "spoken_error": "すきです。", "edit_type": "substitution", "expected_construct_role": ORDINARY_ROLE},
    {"case_id": "fricative_sh_s", "target": "しゃしんです。", "spoken_error": "さしんです。", "edit_type": "substitution", "expected_construct_role": ORDINARY_ROLE},
    {"case_id": "affricate_ch_sh", "target": "ちずです。", "spoken_error": "しずです。", "edit_type": "substitution", "expected_construct_role": ORDINARY_ROLE},
    {"case_id": "sokuon_deletion", "target": "かっこです。", "spoken_error": "かこです。", "edit_type": "deletion", "expected_construct_role": SPECIAL_MORA_ROLE},
    {"case_id": "mora_n_deletion", "target": "みんなです。", "spoken_error": "みなです。", "edit_type": "deletion", "expected_construct_role": SPECIAL_MORA_ROLE},
    {"case_id": "long_vowel_deletion", "target": "おばあさんです。", "spoken_error": "おばさんです。", "edit_type": "deletion", "expected_construct_role": LONG_VOWEL_ROLE},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/controlled_tts_phone_edit_preflight.json")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _target_contract(text: str) -> tuple[list[str], dict[str, Any]]:
    evidence = build_japanese_target_evidence(text)
    phones, _ = sanitize_canonical_phones(evidence.phones)
    role_result = infer_phone_construct_roles(evidence.reading_kana, phones)
    return list(phones), role_result.to_dict()


def _phones(text: str) -> list[str]:
    evidence = build_japanese_target_evidence(text)
    phones, _ = sanitize_canonical_phones(evidence.phones)
    return list(phones)


def _single_edit(target: list[str], spoken: list[str], edit_type: str) -> Dict[str, Any]:
    if edit_type == "substitution":
        if len(target) != len(spoken):
            return {"available": False, "reason": "substitution_phone_count_mismatch"}
        changed = [i for i, (a, b) in enumerate(zip(target, spoken)) if a != b]
        if len(changed) != 1:
            return {"available": False, "reason": "expected_exactly_one_substitution", "changed_indices": changed}
        index = changed[0]
        return {
            "available": True,
            "edit_type": "substitution",
            "target_index": index,
            "target_phone": target[index],
            "error_phone": spoken[index],
            "sequence_edit_ambiguous": False,
        }
    if edit_type == "deletion":
        if len(target) != len(spoken) + 1:
            return {"available": False, "reason": "deletion_phone_count_mismatch"}
        candidates = [i for i in range(len(target)) if target[:i] + target[i + 1 :] == spoken]
        if len(candidates) != 1:
            return {
                "available": False,
                "reason": "expected_exactly_one_deletion",
                "candidate_indices": candidates,
                "sequence_edit_ambiguous": len(candidates) > 1,
            }
        index = candidates[0]
        return {
            "available": True,
            "edit_type": "deletion",
            "target_index": index,
            "target_phone": target[index],
            "error_phone": None,
            "candidate_indices": candidates,
            "sequence_edit_ambiguous": False,
        }
    return {"available": False, "reason": "unsupported_edit_type"}


def _resolve_edit_for_construct(
    edit: Dict[str, Any],
    target_phones: list[str],
    target_roles: Dict[str, Any],
    expected_role: str,
) -> Dict[str, Any]:
    """Resolve only sequence-equivalent deletion ambiguity using target roles."""
    roles = [str(role) for role in (target_roles.get("roles") or [])]
    if not bool(target_roles.get("available")) or len(roles) != len(target_phones):
        return {
            **edit,
            "available": False,
            "reason": "target_construct_roles_unavailable_for_edit_resolution",
        }

    if bool(edit.get("available")):
        index = int(edit.get("target_index", -1))
        if index < 0 or index >= len(roles):
            return {**edit, "available": False, "reason": "edit_index_outside_construct_roles"}
        return {
            **edit,
            "target_construct_role": roles[index],
            "resolved_by_construct_role": False,
        }

    candidates = [int(index) for index in (edit.get("candidate_indices") or [])]
    if str(edit.get("reason")) != "expected_exactly_one_deletion" or len(candidates) <= 1:
        return edit

    matching = [index for index in candidates if 0 <= index < len(roles) and roles[index] == str(expected_role)]
    if len(matching) != 1:
        return {
            **edit,
            "available": False,
            "reason": "ambiguous_deletion_not_resolved_by_construct_role",
            "expected_construct_role": str(expected_role),
            "candidate_construct_roles": {
                str(index): roles[index] if 0 <= index < len(roles) else "out_of_range"
                for index in candidates
            },
        }

    index = matching[0]
    return {
        "available": True,
        "edit_type": "deletion",
        "target_index": index,
        "target_phone": target_phones[index],
        "error_phone": None,
        "candidate_indices": candidates,
        "sequence_edit_ambiguous": True,
        "target_construct_role": roles[index],
        "resolved_by_construct_role": True,
        "resolution_basis": "target_construct_role_not_acoustic_phone_identity",
    }


def _synthesize(text: str) -> np.ndarray:
    waveform, sr = pyopenjtalk.tts(text)
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise RuntimeError("pyopenjtalk returned empty waveform")
    peak = float(np.max(np.abs(values)))
    if peak > 0:
        values = values / peak
    if int(sr) != 16000:
        values = librosa.resample(values, orig_sr=int(sr), target_sr=16000)
    speech, _region = trim_to_speech(values.astype(np.float64), 16000)
    return np.asarray(speech, dtype=np.float32)


def _evaluate(model: BeatriceInfer, waveform: np.ndarray, target_phones: list[str]):
    logits, vocab, blank_id = model.infer(waveform)
    return compute_restricted_fgop_sf_sd_features(
        logits,
        target_phones,
        vocab=vocab,
        blank_id=blank_id,
        model_id=model.model_id,
        revision=model.revision,
    )


def main() -> None:
    args = parse_args()
    model = BeatriceInfer(allow_download=bool(args.allow_download), device=args.device)
    rows: list[Dict[str, Any]] = []

    for case in CASES:
        target_phones, target_roles = _target_contract(case["target"])
        error_phones = _phones(case["spoken_error"])
        raw_edit = _single_edit(target_phones, error_phones, case["edit_type"])
        edit = _resolve_edit_for_construct(
            raw_edit,
            target_phones,
            target_roles,
            str(case["expected_construct_role"]),
        )
        record: Dict[str, Any] = {
            **case,
            "target_phones": target_phones,
            "target_construct_roles": target_roles,
            "spoken_error_phones": error_phones,
            "raw_sequence_edit": raw_edit,
            "edit": edit,
            "synthetic_control_only": True,
            "learner_error_validity": False,
        }
        if not bool(target_roles.get("available")):
            record["status"] = "invalid_target_construct_roles"
            rows.append(record)
            continue
        if not edit.get("available"):
            record["status"] = "invalid_control_definition"
            rows.append(record)
            continue

        index = int(edit["target_index"])
        roles = list(target_roles.get("roles") or [])
        if index >= len(roles):
            record["status"] = "edit_index_outside_construct_roles"
            rows.append(record)
            continue
        target_role = str(roles[index])
        expected_role = str(case["expected_construct_role"])
        record["construct_role_matches_control_design"] = target_role == expected_role
        if target_role != expected_role:
            record["status"] = "construct_role_control_mismatch"
            rows.append(record)
            continue

        correct_audio = _synthesize(case["target"])
        error_audio = _synthesize(case["spoken_error"])
        correct = _evaluate(model, correct_audio, target_phones)
        erroneous = _evaluate(model, error_audio, target_phones)
        if not correct.available or not erroneous.available:
            record.update(
                {
                    "status": "feature_unavailable",
                    "correct_available": correct.available,
                    "error_available": erroneous.available,
                    "correct_warnings": correct.warnings,
                    "error_warnings": erroneous.warnings,
                }
            )
            rows.append(record)
            continue

        correct_row = correct.rows[index]
        error_row = erroneous.rows[index]
        if edit["edit_type"] == "substitution":
            if target_role != ORDINARY_ROLE:
                record["status"] = "substitution_control_on_nonordinary_construct"
                rows.append(record)
                continue
            error_phone = str(edit["error_phone"])
            candidate_present = error_phone in error_row.substitution_lprs
            correct_specific = correct_row.substitution_lprs.get(error_phone)
            error_specific = error_row.substitution_lprs.get(error_phone)
            feature_interpretation = "target_specific_substitution_lpr"
        else:
            error_phone = None
            candidate_present = True
            correct_specific = correct_row.deletion_lpr
            error_specific = error_row.deletion_lpr
            feature_interpretation = (
                "long_vowel_extension_deletion_support_lpr"
                if target_role == LONG_VOWEL_ROLE
                else "special_mora_deletion_support_lpr"
                if target_role == SPECIAL_MORA_ROLE
                else "ordinary_phone_deletion_support_lpr"
            )

        localized_direction_pass = (
            candidate_present
            and correct_specific is not None
            and error_specific is not None
            and float(error_specific) < float(correct_specific)
        )
        strong_sign_flip = (
            localized_direction_pass
            and float(correct_specific) > 0.0
            and float(error_specific) < 0.0
        )
        record.update(
            {
                "status": "evaluated",
                "target_construct_role": target_role,
                "target_specific_feature_interpretation": feature_interpretation,
                "candidate_present": candidate_present,
                "correct_target_specific_lpr": correct_specific,
                "error_target_specific_lpr": error_specific,
                "target_specific_lpr_delta_error_minus_correct": (
                    float(error_specific) - float(correct_specific)
                    if correct_specific is not None and error_specific is not None else None
                ),
                "localized_direction_pass": localized_direction_pass,
                "strong_sign_flip": strong_sign_flip,
                "correct_best_noncanonical_type": correct_row.best_noncanonical_type,
                "correct_best_noncanonical_phone": correct_row.best_noncanonical_phone,
                "error_best_noncanonical_type": error_row.best_noncanonical_type,
                "error_best_noncanonical_phone": error_row.best_noncanonical_phone,
                "search_policy": error_row.search_policy,
                "candidate_phones": list(error_row.candidate_phones),
                "best_noncanonical_is_direct_correctness_label": False,
            }
        )
        rows.append(record)

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    payload = {
        "schema": "controlled_tts_phone_edit_preflight_v3",
        "model_id": model.model_id,
        "revision": model.revision,
        "tts_backend": "pyopenjtalk_plus_hts",
        "generated_audio_committed": False,
        "generated_audio_uploaded": False,
        "synthetic_control_is_learner_validity": False,
        "construct_roles_checked_before_acoustic_control": True,
        "sequence_equivalent_deletion_can_be_resolved_by_target_construct_role": True,
        "long_vowel_extension_is_ordinary_segmental_clarity": False,
        "score_mapped": False,
        "product_calibrated": False,
        "product_score_changed": False,
        "human_recording_gate_changed": False,
        "rows": rows,
        "summary": {
            "case_count": len(rows),
            "evaluated_count": len(evaluated),
            "ordinary_segmental_control_count": sum(row.get("target_construct_role") == ORDINARY_ROLE for row in evaluated),
            "special_mora_control_count": sum(row.get("target_construct_role") == SPECIAL_MORA_ROLE for row in evaluated),
            "long_vowel_control_count": sum(row.get("target_construct_role") == LONG_VOWEL_ROLE for row in evaluated),
            "sequence_ambiguous_controls_resolved_by_construct_role": sum(bool(row.get("edit", {}).get("resolved_by_construct_role")) for row in evaluated),
            "localized_direction_pass_count": sum(bool(row.get("localized_direction_pass")) for row in evaluated),
            "strong_sign_flip_count": sum(bool(row.get("strong_sign_flip")) for row in evaluated),
            "all_candidates_present": all(bool(row.get("candidate_present")) for row in evaluated) if evaluated else False,
            "all_construct_roles_match_control_design": all(bool(row.get("construct_role_matches_control_design")) for row in evaluated) if evaluated else False,
            "interpretation": "construct_audited_synthetic_localization_sanity_only_not_Japanese_L2_pronunciation_validity",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print("HUMAN RECORDING GATE: UNCHANGED / BLOCKED")
    print("PRODUCT SCORE: UNCHANGED / SYNTHETIC RESEARCH CONTROL ONLY")


if __name__ == "__main__":
    main()

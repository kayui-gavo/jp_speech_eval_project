#!/usr/bin/env python3
"""Compare controlled local phone edits across three pinned Japanese CTC models.

The existing Beatrice-only synthetic control established directional sanity but
also revealed that a correct ``つきです`` token could already have a negative
``ts``-vs-``s`` LPR. This script asks whether such behavior is model-specific or
shared across independent Japanese phone-CTC backbones.

Generated OpenJTalk audio is ephemeral. Results are research diagnostics only;
no sign threshold is treated as a pronunciation decision.
"""

from __future__ import annotations

import argparse
import gc
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

from jp_speech_eval.dual_ctc_phone_candidate import (  # noqa: E402
    DISTILHUBERT_DUAL_CTC_MODEL,
    WAVLM_DUAL_CTC_MODEL,
    DualCtcPhoneCandidateBackend,
)
from jp_speech_eval.japanese_phoneme_gop import (  # noqa: E402
    DEFAULT_PHONE_CTC_MODEL,
    DEFAULT_PHONE_CTC_REVISION,
    sanitize_canonical_phones,
)
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.restricted_segmentation_free_gop import compute_restricted_fgop_sf_sd_features  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402
from run_official_jvs_phone_ctc_anchor_preflight import BeatriceInfer  # noqa: E402


DISTIL_REVISION = "01ffc3e5b0e49ba34180d50c48ea4111aa041cfd"
WAVLM_REVISION = "47fa985035342365bcec4948bd821aaf58dd778a"

CASES = (
    {"case_id": "voicing_b_p", "target": "バスです。", "spoken_error": "パスです。", "edit_type": "substitution"},
    {"case_id": "voicing_g_k", "target": "かぎです。", "spoken_error": "かきです。", "edit_type": "substitution"},
    {"case_id": "affricate_ts_s", "target": "つきです。", "spoken_error": "すきです。", "edit_type": "substitution"},
    {"case_id": "fricative_sh_s", "target": "しゃしんです。", "spoken_error": "さしんです。", "edit_type": "substitution"},
    {"case_id": "affricate_ch_sh", "target": "ちずです。", "spoken_error": "しずです。", "edit_type": "substitution"},
    {"case_id": "sokuon_deletion", "target": "かっこです。", "spoken_error": "かこです。", "edit_type": "deletion"},
    {"case_id": "mora_n_deletion", "target": "みんなです。", "spoken_error": "みなです。", "edit_type": "deletion"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/multibackbone_controlled_phone_edits.json")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


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
        i = changed[0]
        return {"available": True, "edit_type": "substitution", "target_index": i, "target_phone": target[i], "error_phone": spoken[i]}
    if edit_type == "deletion":
        if len(target) != len(spoken) + 1:
            return {"available": False, "reason": "deletion_phone_count_mismatch"}
        indices = [i for i in range(len(target)) if target[:i] + target[i + 1 :] == spoken]
        if len(indices) != 1:
            return {"available": False, "reason": "expected_exactly_one_deletion", "candidate_indices": indices}
        i = indices[0]
        return {"available": True, "edit_type": "deletion", "target_index": i, "target_phone": target[i], "error_phone": None}
    return {"available": False, "reason": "unsupported_edit_type"}


def _tts(text: str) -> np.ndarray:
    waveform, sr = pyopenjtalk.tts(text)
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak > 0:
        values = values / peak
    if int(sr) != 16000:
        values = librosa.resample(values, orig_sr=int(sr), target_sr=16000)
    speech, _region = trim_to_speech(values.astype(np.float64), 16000)
    return np.asarray(speech, dtype=np.float32)


def _prepared_cases() -> list[Dict[str, Any]]:
    rows = []
    for case in CASES:
        target_phones = _phones(case["target"])
        error_phones = _phones(case["spoken_error"])
        rows.append(
            {
                **case,
                "target_phones": target_phones,
                "error_phones": error_phones,
                "edit": _single_edit(target_phones, error_phones, case["edit_type"]),
                "correct_audio": _tts(case["target"]),
                "error_audio": _tts(case["spoken_error"]),
            }
        )
    return rows


class _DualInfer:
    def __init__(self, model_id: str, revision: str, *, allow_download: bool, device: str | None) -> None:
        self.backend = DualCtcPhoneCandidateBackend(
            model_id=model_id,
            revision=revision,
            device=device,
            local_files_only=not allow_download,
        )
        self.model_id = model_id
        self.revision = revision

    def infer(self, waveform: np.ndarray):
        logits, vocab, blank_id, _ = self.backend.infer_logical_phone_logits(waveform, sr=16000)
        return logits, vocab, int(blank_id)

    def close(self) -> None:
        self.backend.processor = None
        self.backend.model = None
        self.backend.tokenizer = None
        gc.collect()


def _models(*, allow_download: bool, device: str | None):
    return [
        (
            "beatrice",
            BeatriceInfer(allow_download=allow_download, device=device),
        ),
        (
            "distilhubert_dual_ctc",
            _DualInfer(DISTILHUBERT_DUAL_CTC_MODEL, DISTIL_REVISION, allow_download=allow_download, device=device),
        ),
        (
            "wavlm_dual_ctc",
            _DualInfer(WAVLM_DUAL_CTC_MODEL, WAVLM_REVISION, allow_download=allow_download, device=device),
        ),
    ]


def _feature(model, audio: np.ndarray, phones: list[str]):
    logits, vocab, blank_id = model.infer(audio)
    return compute_restricted_fgop_sf_sd_features(
        logits,
        phones,
        vocab=vocab,
        blank_id=blank_id,
        model_id=model.model_id,
        revision=model.revision,
    )


def main() -> None:
    args = parse_args()
    cases = _prepared_cases()
    model_results = []
    for model_key, model in _models(allow_download=bool(args.allow_download), device=args.device):
        rows = []
        try:
            for case in cases:
                edit = case["edit"]
                record: Dict[str, Any] = {
                    key: case[key]
                    for key in ("case_id", "target", "spoken_error", "edit_type", "target_phones", "error_phones")
                }
                record["edit"] = edit
                if not edit.get("available"):
                    record["status"] = "invalid_control_definition"
                    rows.append(record)
                    continue
                correct = _feature(model, case["correct_audio"], case["target_phones"])
                erroneous = _feature(model, case["error_audio"], case["target_phones"])
                index = int(edit["target_index"])
                if not correct.available or not erroneous.available:
                    record.update({"status": "feature_unavailable", "correct": correct.to_dict(), "error": erroneous.to_dict()})
                    rows.append(record)
                    continue
                c = correct.rows[index]
                e = erroneous.rows[index]
                if edit["edit_type"] == "substitution":
                    alt = str(edit["error_phone"])
                    c_lpr = c.substitution_lprs.get(alt)
                    e_lpr = e.substitution_lprs.get(alt)
                    candidate_present = alt in e.substitution_lprs
                else:
                    alt = None
                    c_lpr = c.deletion_lpr
                    e_lpr = e.deletion_lpr
                    candidate_present = True
                direction = (
                    candidate_present
                    and c_lpr is not None
                    and e_lpr is not None
                    and float(e_lpr) < float(c_lpr)
                )
                record.update(
                    {
                        "status": "evaluated",
                        "candidate_present": candidate_present,
                        "correct_target_specific_lpr": c_lpr,
                        "error_target_specific_lpr": e_lpr,
                        "delta_error_minus_correct": (
                            float(e_lpr) - float(c_lpr)
                            if c_lpr is not None and e_lpr is not None else None
                        ),
                        "localized_direction_pass": direction,
                        "correct_sign_positive": bool(c_lpr is not None and float(c_lpr) > 0),
                        "error_sign_negative": bool(e_lpr is not None and float(e_lpr) < 0),
                        "strong_sign_flip": bool(
                            direction
                            and c_lpr is not None
                            and e_lpr is not None
                            and float(c_lpr) > 0
                            and float(e_lpr) < 0
                        ),
                    }
                )
                rows.append(record)
        finally:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        evaluated = [row for row in rows if row.get("status") == "evaluated"]
        model_results.append(
            {
                "model_key": model_key,
                "model_id": model.model_id,
                "revision": model.revision,
                "rows": rows,
                "summary": {
                    "evaluated_count": len(evaluated),
                    "localized_direction_pass_count": sum(bool(row.get("localized_direction_pass")) for row in evaluated),
                    "correct_sign_positive_count": sum(bool(row.get("correct_sign_positive")) for row in evaluated),
                    "strong_sign_flip_count": sum(bool(row.get("strong_sign_flip")) for row in evaluated),
                },
            }
        )

    by_case: Dict[str, Any] = {}
    for case in CASES:
        entries = []
        for model in model_results:
            row = next(row for row in model["rows"] if row["case_id"] == case["case_id"])
            entries.append(
                {
                    "model_key": model["model_key"],
                    "correct_lpr": row.get("correct_target_specific_lpr"),
                    "error_lpr": row.get("error_target_specific_lpr"),
                    "direction_pass": row.get("localized_direction_pass"),
                    "correct_sign_positive": row.get("correct_sign_positive"),
                    "strong_sign_flip": row.get("strong_sign_flip"),
                }
            )
        by_case[case["case_id"]] = {
            "models": entries,
            "direction_consensus_count": sum(bool(entry.get("direction_pass")) for entry in entries),
            "correct_positive_consensus_count": sum(bool(entry.get("correct_sign_positive")) for entry in entries),
        }

    payload = {
        "schema": "multibackbone_controlled_phone_edits_v1",
        "tts_backend": "pyopenjtalk_plus_hts",
        "generated_audio_committed": False,
        "generated_audio_uploaded": False,
        "synthetic_control_is_learner_validity": False,
        "global_lpr_zero_threshold_validated": False,
        "score_mapped": False,
        "product_calibrated": False,
        "product_score_changed": False,
        "human_recording_gate_changed": False,
        "models": model_results,
        "cross_model_case_summary": by_case,
        "interpretation": (
            "paired local directionality and model-consensus audit; raw LPR scales are "
            "model-specific and are never averaged"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    for model in model_results:
        print(model["model_key"], model["summary"])
    print("GLOBAL LPR<0 ERROR THRESHOLD: NOT VALIDATED")
    print("HUMAN RECORDING GATE: UNCHANGED / BLOCKED")
    print("PRODUCT SCORE: UNCHANGED / RESEARCH ONLY")


if __name__ == "__main__":
    main()

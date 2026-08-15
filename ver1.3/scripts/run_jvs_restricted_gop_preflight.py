#!/usr/bin/env python3
"""Run Japanese restricted-substitution GOP features on official JVS anchors.

This is an automatic native false-alarm pressure test. It does *not* infer that
a negative local margin is a pronunciation error.

The JVS target uses the manifest's explicit reviewed logical-phone sequence and
per-phone construct roles. Neither surface-kanji G2P nor re-G2P of the kana
reading is accepted as the phone target: Stage-0 found both paths could alter
``明王 / みょうおう`` and create repeated pseudo-errors across native speakers.

Every reported negative phone position is mapped back to the reviewed
surface/reading segment and construct role. Ordinary segmental false-alarm
statistics exclude both special morae and long-vowel extension morae. For a
long-vowel extension, generic vowel-substitution wins are not interpreted as
clarity evidence; only the canonical-vs-deletion LPR is summarized as timing-
support pressure until a construct-specific timing criterion is validated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from jp_speech_eval.audio_features import load_audio  # noqa: E402
from jp_speech_eval.japanese_phone_roles import (  # noqa: E402
    LONG_VOWEL_ROLE,
    ORDINARY_ROLE,
    SPECIAL_MORA_ROLE,
)
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.japanese_phoneme_gop import sanitize_canonical_phones, segmental_competitor_ids  # noqa: E402
from jp_speech_eval.restricted_segmentation_free_gop import compute_restricted_fgop_sf_sd_features  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402
from run_official_jvs_phone_ctc_anchor_preflight import (  # noqa: E402
    BeatriceInfer,
    TARGET_TEXT,
    _load_manifest,
    _reviewed_roles,
    _reviewed_target,
    _source_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="outputs/jvs_restricted_gop_preflight.json")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _reviewed_index_metadata(
    rows: Sequence[Dict[str, Any]],
    reviewed_phones: Sequence[str],
    reviewed_roles: Sequence[str],
) -> list[Dict[str, Any]]:
    serialized = [
        json.dumps(row.get("target_phone_index_metadata") or [], ensure_ascii=False, sort_keys=True)
        for row in rows
    ]
    if not serialized or any(value != serialized[0] for value in serialized[1:]):
        raise ValueError("JVS manifest target_phone_index_metadata differs across speakers")
    metadata = list(rows[0].get("target_phone_index_metadata") or [])
    if len(metadata) != len(reviewed_phones) or len(reviewed_roles) != len(reviewed_phones):
        raise ValueError("JVS phone-index metadata/role length does not match reviewed phone sequence")
    normalized: list[Dict[str, Any]] = []
    for index, (meta, phone, role) in enumerate(zip(metadata, reviewed_phones, reviewed_roles)):
        if int(meta.get("phone_index", -1)) != index:
            raise ValueError("JVS phone-index metadata is not contiguous")
        if str(meta.get("phone") or "") != str(phone):
            raise ValueError("JVS phone-index metadata phone does not match reviewed target")
        if str(meta.get("construct_role") or "") != str(role):
            raise ValueError("JVS phone-index metadata construct role does not match reviewed target")
        normalized.append(
            {
                "phone_index": index,
                "phone": str(phone),
                "construct_role": str(role),
                "segment_index": int(meta.get("segment_index", -1)),
                "segment_phone_index": int(meta.get("segment_phone_index", -1)),
                "segment_surface": str(meta.get("segment_surface") or ""),
                "segment_reading": str(meta.get("segment_reading") or ""),
            }
        )
    return normalized


def _negative_position(row, meta: Dict[str, Any], *, criterion: str) -> Dict[str, Any]:
    return {
        "phone_index": int(row.phone_index),
        "canonical_phone": str(row.canonical_phone),
        "target_construct_role": str(meta["construct_role"]),
        "target_segment_index": meta["segment_index"],
        "target_segment_phone_index": meta["segment_phone_index"],
        "target_segment_surface": meta["segment_surface"],
        "target_segment_reading": meta["segment_reading"],
        "criterion": criterion,
        "best_noncanonical_type": row.best_noncanonical_type,
        "best_noncanonical_phone": row.best_noncanonical_phone,
        "best_noncanonical_lpr": float(row.best_noncanonical_lpr),
        "deletion_lpr": float(row.deletion_lpr),
        "candidate_count": int(row.candidate_count),
        "search_policy": row.search_policy,
        "negative_margin_is_pronunciation_error": False,
    }


def main() -> None:
    args = parse_args()
    sample_rows = _load_manifest(Path(args.manifest))
    target_reading, reviewed_phones = _reviewed_target(sample_rows)
    reviewed_roles = _reviewed_roles(sample_rows, reviewed_phones)
    index_metadata = _reviewed_index_metadata(sample_rows, reviewed_phones, reviewed_roles)
    target = build_japanese_target_evidence(
        TARGET_TEXT,
        reading_override=target_reading,
        phones_override=reviewed_phones,
    )
    phones, dropped = sanitize_canonical_phones(target.phones)
    if dropped or phones != reviewed_phones:
        raise RuntimeError("reviewed JVS phone override changed during target sanitization")
    model = BeatriceInfer(allow_download=bool(args.allow_download), device=args.device)

    rows: list[Dict[str, Any]] = []
    for sample in sample_rows:
        audio = load_audio(str(Path(sample["path"])), sr=16000)
        speech, region = trim_to_speech(audio.y, audio.sr)
        logits, vocab, blank_id = model.infer(np.asarray(speech, dtype=np.float32))
        restricted = compute_restricted_fgop_sf_sd_features(
            logits,
            phones,
            vocab=vocab,
            blank_id=blank_id,
            model_id=model.model_id,
            revision=model.revision,
        )
        unrestricted_count = len(segmental_competitor_ids(vocab, blank_id=blank_id))
        if not restricted.available:
            rows.append(
                {
                    "speaker": sample["speaker"],
                    "status": "unavailable",
                    "source_provenance": _source_provenance(sample),
                    "speech_region": region.to_dict(),
                    "restricted": restricted.to_dict(),
                }
            )
            continue

        ordinary_rows = [
            row for row in restricted.rows
            if index_metadata[int(row.phone_index)]["construct_role"] == ORDINARY_ROLE
        ]
        special_rows = [
            row for row in restricted.rows
            if index_metadata[int(row.phone_index)]["construct_role"] == SPECIAL_MORA_ROLE
        ]
        long_rows = [
            row for row in restricted.rows
            if index_metadata[int(row.phone_index)]["construct_role"] == LONG_VOWEL_ROLE
        ]

        # Ordinary clarity uses the best restricted noncanonical alternative.
        ordinary_negative = [row for row in ordinary_rows if row.best_noncanonical_lpr < 0.0]
        # Timing constructs use deletion support only. A generic vowel
        # substitution at a long-vowel extension is not a clarity event.
        special_deletion_negative = [row for row in special_rows if row.deletion_lpr < 0.0]
        long_deletion_negative = [row for row in long_rows if row.deletion_lpr < 0.0]

        negative_positions: list[Dict[str, Any]] = []
        for row in ordinary_negative:
            negative_positions.append(
                _negative_position(
                    row,
                    index_metadata[int(row.phone_index)],
                    criterion="ordinary_best_restricted_noncanonical_lpr_lt_0",
                )
            )
        for row in special_deletion_negative:
            negative_positions.append(
                _negative_position(
                    row,
                    index_metadata[int(row.phone_index)],
                    criterion="special_mora_deletion_lpr_lt_0",
                )
            )
        for row in long_deletion_negative:
            negative_positions.append(
                _negative_position(
                    row,
                    index_metadata[int(row.phone_index)],
                    criterion="long_vowel_extension_deletion_lpr_lt_0",
                )
            )

        raw_best_negative_count = sum(row.best_noncanonical_lpr < 0.0 for row in restricted.rows)
        rows.append(
            {
                "speaker": sample["speaker"],
                "status": "evaluated",
                "source_provenance": _source_provenance(sample),
                "speech_region": region.to_dict(),
                "phone_count": len(restricted.rows),
                "ordinary_segmental_phone_count": len(ordinary_rows),
                "special_mora_count": len(special_rows),
                "long_vowel_timing_phone_count": len(long_rows),
                "unrestricted_segmental_candidate_count_per_position": unrestricted_count,
                "restricted_candidate_count_mean": restricted.summary["candidate_count_mean"],
                "restricted_candidate_ratio_vs_unrestricted": (
                    float(restricted.summary["candidate_count_mean"]) / unrestricted_count
                    if unrestricted_count > 0 else None
                ),
                "raw_best_noncanonical_outscore_count_all_constructs": raw_best_negative_count,
                "raw_best_noncanonical_outscore_rate_all_constructs": raw_best_negative_count / len(restricted.rows),
                "ordinary_segmental_noncanonical_outscore_count": len(ordinary_negative),
                "ordinary_segmental_noncanonical_outscore_rate": (
                    len(ordinary_negative) / len(ordinary_rows) if ordinary_rows else None
                ),
                "special_mora_deletion_outscore_count": len(special_deletion_negative),
                "special_mora_deletion_outscore_rate": (
                    len(special_deletion_negative) / len(special_rows) if special_rows else None
                ),
                "long_vowel_deletion_outscore_count": len(long_deletion_negative),
                "long_vowel_deletion_outscore_rate": (
                    len(long_deletion_negative) / len(long_rows) if long_rows else None
                ),
                "long_vowel_generic_substitution_is_clarity_evidence": False,
                "fallback_position_count": restricted.summary["fallback_position_count"],
                "construct_relevant_negative_positions": negative_positions,
                "restricted_summary": restricted.summary,
            }
        )

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    ordinary_rates = [
        float(row["ordinary_segmental_noncanonical_outscore_rate"])
        for row in evaluated
        if row.get("ordinary_segmental_noncanonical_outscore_rate") is not None
    ]
    long_deletion_rates = [
        float(row["long_vowel_deletion_outscore_rate"])
        for row in evaluated
        if row.get("long_vowel_deletion_outscore_rate") is not None
    ]
    candidate_ratios = [
        float(row["restricted_candidate_ratio_vs_unrestricted"])
        for row in evaluated
        if row.get("restricted_candidate_ratio_vs_unrestricted") is not None
    ]
    payload = {
        "schema": "jvs_restricted_gop_native_preflight_v5",
        "target_text": TARGET_TEXT,
        "target_reading": target_reading,
        "target_reading_source": "reviewed_manifest_override",
        "target_phone_source": "reviewed_logical_phone_override_v1",
        "automatic_surface_g2p_used_for_phone_target": False,
        "automatic_kana_g2p_used_for_phone_target": False,
        "target_phones": phones,
        "target_phone_roles": reviewed_roles,
        "target_phone_index_metadata": index_metadata,
        "dropped_nonsegmental_target_tokens": dropped,
        "model_id": model.model_id,
        "revision": model.revision,
        "native_audio_has_phone_error_labels": False,
        "negative_margin_is_pronunciation_error": False,
        "long_vowel_extension_is_ordinary_segmental_clarity": False,
        "long_vowel_generic_substitution_is_clarity_evidence": False,
        "score_mapped": False,
        "product_calibrated": False,
        "product_score_changed": False,
        "human_recording_gate_changed": False,
        "rows": rows,
        "summary": {
            "evaluated_speakers": len(evaluated),
            "ordinary_segmental_noncanonical_outscore_rate_mean": (
                statistics.mean(ordinary_rates) if ordinary_rates else None
            ),
            "ordinary_segmental_noncanonical_outscore_rate_max": (
                max(ordinary_rates) if ordinary_rates else None
            ),
            "long_vowel_deletion_outscore_rate_mean": (
                statistics.mean(long_deletion_rates) if long_deletion_rates else None
            ),
            "restricted_candidate_ratio_vs_unrestricted_mean": (
                statistics.mean(candidate_ratios) if candidate_ratios else None
            ),
            "target_reading_reviewed_before_phone_scoring": True,
            "target_phone_sequence_explicitly_reviewed": True,
            "target_phone_construct_roles_explicitly_reviewed": True,
            "negative_positions_are_segment_and_construct_annotated": True,
            "ordinary_false_alarm_denominator_excludes_timing_constructs": True,
            "long_vowel_timing_uses_deletion_support_only_in_this_summary": True,
            "text_frontend_g2p_is_authoritative_phone_source": False,
            "interpretation": "construct_stratified_native_false_alarm_pressure_test_not_pronunciation_validity",
            "required_next_step": "compare_construct_specific_features_on_labeled_or_controlled_learner_errors_before_any_clarity_or_timing_mapping",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print("reviewed target phone count:", len(phones))
    print("reviewed long-vowel timing phone count:", sum(role == LONG_VOWEL_ROLE for role in reviewed_roles))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print("HUMAN RECORDING GATE: UNCHANGED / BLOCKED")
    print("PRODUCT SCORE: UNCHANGED / RESEARCH FEATURE ONLY")


if __name__ == "__main__":
    main()

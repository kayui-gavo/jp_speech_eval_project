#!/usr/bin/env python3
"""Run Japanese restricted-substitution GOP features on official JVS anchors.

This is an automatic native false-alarm pressure test. It does *not* infer that
a negative local margin is a pronunciation error.

The JVS target uses the manifest's explicit reviewed logical-phone sequence.
Neither surface-kanji G2P nor re-G2P of the kana reading is accepted as the
phone target: Stage-0 found both paths could alter ``明王 / みょうおう`` and
create repeated pseudo-errors across native speakers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from jp_speech_eval.audio_features import load_audio  # noqa: E402
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.japanese_phoneme_gop import sanitize_canonical_phones, segmental_competitor_ids  # noqa: E402
from jp_speech_eval.restricted_segmentation_free_gop import compute_restricted_fgop_sf_sd_features  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402
from run_official_jvs_phone_ctc_anchor_preflight import (  # noqa: E402
    BeatriceInfer,
    TARGET_TEXT,
    _load_manifest,
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


def main() -> None:
    args = parse_args()
    sample_rows = _load_manifest(Path(args.manifest))
    target_reading, reviewed_phones = _reviewed_target(sample_rows)
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

        negative = [row for row in restricted.rows if row.best_noncanonical_lpr < 0.0]
        segmental_rows = [row for row in restricted.rows if row.canonical_phone not in {"N", "cl"}]
        special_rows = [row for row in restricted.rows if row.canonical_phone in {"N", "cl"}]
        rows.append(
            {
                "speaker": sample["speaker"],
                "status": "evaluated",
                "source_provenance": _source_provenance(sample),
                "speech_region": region.to_dict(),
                "phone_count": len(restricted.rows),
                "unrestricted_segmental_candidate_count_per_position": unrestricted_count,
                "restricted_candidate_count_mean": restricted.summary["candidate_count_mean"],
                "restricted_candidate_ratio_vs_unrestricted": (
                    float(restricted.summary["candidate_count_mean"]) / unrestricted_count
                    if unrestricted_count > 0 else None
                ),
                "restricted_noncanonical_outscore_count": len(negative),
                "restricted_noncanonical_outscore_rate": len(negative) / len(restricted.rows),
                "segmental_noncanonical_outscore_count": sum(row.best_noncanonical_lpr < 0 for row in segmental_rows),
                "segmental_phone_count": len(segmental_rows),
                "special_mora_negative_count": sum(row.best_noncanonical_lpr < 0 for row in special_rows),
                "special_mora_count": len(special_rows),
                "fallback_position_count": restricted.summary["fallback_position_count"],
                "negative_positions": [
                    {
                        "phone_index": row.phone_index,
                        "canonical_phone": row.canonical_phone,
                        "best_noncanonical_type": row.best_noncanonical_type,
                        "best_noncanonical_phone": row.best_noncanonical_phone,
                        "best_noncanonical_lpr": row.best_noncanonical_lpr,
                        "candidate_count": row.candidate_count,
                        "search_policy": row.search_policy,
                    }
                    for row in negative
                ],
                "restricted_summary": restricted.summary,
            }
        )

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    native_rates = [float(row["restricted_noncanonical_outscore_rate"]) for row in evaluated]
    candidate_ratios = [
        float(row["restricted_candidate_ratio_vs_unrestricted"])
        for row in evaluated
        if row.get("restricted_candidate_ratio_vs_unrestricted") is not None
    ]
    payload = {
        "schema": "jvs_restricted_gop_native_preflight_v3",
        "target_text": TARGET_TEXT,
        "target_reading": target_reading,
        "target_reading_source": "reviewed_manifest_override",
        "target_phone_source": "reviewed_logical_phone_override_v1",
        "automatic_surface_g2p_used_for_phone_target": False,
        "automatic_kana_g2p_used_for_phone_target": False,
        "target_phones": phones,
        "dropped_nonsegmental_target_tokens": dropped,
        "model_id": model.model_id,
        "revision": model.revision,
        "native_audio_has_phone_error_labels": False,
        "negative_margin_is_pronunciation_error": False,
        "score_mapped": False,
        "product_calibrated": False,
        "product_score_changed": False,
        "human_recording_gate_changed": False,
        "rows": rows,
        "summary": {
            "evaluated_speakers": len(evaluated),
            "native_noncanonical_outscore_rate_mean": statistics.mean(native_rates) if native_rates else None,
            "native_noncanonical_outscore_rate_max": max(native_rates) if native_rates else None,
            "restricted_candidate_ratio_vs_unrestricted_mean": statistics.mean(candidate_ratios) if candidate_ratios else None,
            "target_reading_reviewed_before_phone_scoring": True,
            "target_phone_sequence_explicitly_reviewed": True,
            "text_frontend_g2p_is_authoritative_phone_source": False,
            "interpretation": "native_false_alarm_pressure_test_not_pronunciation_validity",
            "required_next_step": "compare_on_labeled_or_controlled_learner_errors_before_any_clarity_mapping",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print("reviewed target phone count:", len(phones))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print("HUMAN RECORDING GATE: UNCHANGED / BLOCKED")
    print("PRODUCT SCORE: UNCHANGED / RESEARCH FEATURE ONLY")


if __name__ == "__main__":
    main()

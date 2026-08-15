#!/usr/bin/env python3
"""Run phone-CTC research diagnostics on public UME-JRF learner samples.

The NII-SRC page publishes five test-listening WAVs by native speakers of
Chinese. These samples do not expose their four-teacher grades on the public
web page, so this script treats them as **unlabeled learner-domain anchors**.
It must not infer that any negative phone feature is a learner error.

For the published minimal pair ``じぶつ / じんぶつ`` the script additionally
compares the page target with its partner on the same waveform. The sign is a
content/contrast diagnostic, not pronunciation ground truth.

CTC posterior peakiness uses the full acoustic phone inventory, including
special morae N/cl. This is intentionally broader than the ordinary clarity
competitor set; otherwise the N-bearing minimal pair would under-count phone
probability mass in the model diagnostic itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from jp_speech_eval.ctc_posterior_diagnostics import compute_ctc_posterior_diagnostics  # noqa: E402
from jp_speech_eval.japanese_phone_inventory import (  # noqa: E402
    acoustic_phone_token_ids,
    inventory_semantics,
)
from jp_speech_eval.japanese_phoneme_gop import sanitize_canonical_phones  # noqa: E402
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.restricted_segmentation_free_gop import compute_restricted_fgop_sf_sd_features  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402
from run_official_jvs_phone_ctc_anchor_preflight import BeatriceInfer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="outputs/ume_jrf_public_phone_preflight.json")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _load_manifest(path: Path) -> list[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ume_jrf_public_samples_manifest_v1":
        raise ValueError("unexpected UME-JRF public sample manifest schema")
    rows = list(payload.get("samples") or [])
    if not rows:
        raise ValueError("UME-JRF public sample manifest contains no samples")
    return rows


def _phones(text: str) -> list[str]:
    target = build_japanese_target_evidence(text)
    phones, _dropped = sanitize_canonical_phones(target.phones)
    return list(phones)


def _evaluate_target(model: BeatriceInfer, logits: np.ndarray, vocab: dict[str, int], blank_id: int, text: str):
    phones = _phones(text)
    result = compute_restricted_fgop_sf_sd_features(
        logits,
        phones,
        vocab=vocab,
        blank_id=blank_id,
        model_id=model.model_id,
        revision=model.revision,
    )
    return phones, result


def main() -> None:
    args = parse_args()
    samples = _load_manifest(Path(args.manifest))
    model = BeatriceInfer(allow_download=bool(args.allow_download), device=args.device)
    rows: list[Dict[str, Any]] = []

    for sample in samples:
        audio = load_audio(str(Path(sample["path"])), sr=16000)
        speech, region = trim_to_speech(audio.y, audio.sr)
        logits, vocab, blank_id = model.infer(np.asarray(speech, dtype=np.float32))
        target_phones, target = _evaluate_target(
            model, logits, vocab, blank_id, str(sample["target_text"])
        )
        posterior = compute_ctc_posterior_diagnostics(
            logits,
            blank_id=blank_id,
            phone_token_ids=acoustic_phone_token_ids(vocab, blank_id=blank_id),
        )
        posterior_payload = posterior.to_dict()
        posterior_payload["phone_inventory_semantics"] = inventory_semantics(
            vocab,
            blank_id=blank_id,
        )

        record: Dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "target_text": sample["target_text"],
            "speaker_l1": sample.get("speaker_l1"),
            "source_url": sample.get("source_url"),
            "raw_sha256": sample.get("raw_sha256"),
            "speech_region": region.to_dict(),
            "target_phones": target_phones,
            "target_feature": target.to_dict(),
            "ctc_posterior_diagnostics": posterior_payload,
            "teacher_grade_available_for_public_sample": False,
            "phone_error_labels_available": False,
            "negative_margin_is_pronunciation_error": False,
        }
        if target.available:
            negative = [row for row in target.rows if row.best_noncanonical_lpr < 0.0]
            record["descriptive_noncanonical_outscore_count"] = len(negative)
            record["descriptive_noncanonical_outscore_rate"] = len(negative) / len(target.rows)
            record["fallback_position_count"] = target.summary.get("fallback_position_count", 0)

        partner = sample.get("minimal_pair_partner")
        if partner:
            partner_phones, partner_result = _evaluate_target(
                model, logits, vocab, blank_id, str(partner)
            )
            target_lp = target.summary.get("canonical_ctc_log_posterior") if target.available else None
            partner_lp = (
                partner_result.summary.get("canonical_ctc_log_posterior")
                if partner_result.available else None
            )
            record["minimal_pair_diagnostic"] = {
                "partner_text": partner,
                "partner_phones": partner_phones,
                "partner_feature": partner_result.to_dict(),
                "page_target_minus_partner_sequence_logposterior": (
                    float(target_lp) - float(partner_lp)
                    if target_lp is not None and partner_lp is not None else None
                ),
                "positive_sign_is_pronunciation_correctness": False,
                "interpretation": (
                    "same-waveform minimal-pair sequence preference; ambiguity may reflect "
                    "learner realization, backend behavior, or both"
                ),
            }
        rows.append(record)

    pair_rows = [row for row in rows if "minimal_pair_diagnostic" in row]
    payload = {
        "schema": "ume_jrf_public_phone_preflight_v2",
        "corpus": "UME-JRF",
        "source": "official_NII_SRC_public_test_listening_samples",
        "speaker_domain": "Japanese_L2_speech_native_speakers_of_Chinese",
        "research_use_only": True,
        "public_samples_have_teacher_grades": False,
        "full_corpus_has_four_native_teacher_grading_lists": True,
        "model_id": model.model_id,
        "revision": model.revision,
        "sample_count": len(rows),
        "minimal_pair_sample_count": len(pair_rows),
        "ctc_peakiness_phone_inventory_includes_special_morae": True,
        "score_mapped": False,
        "product_calibrated": False,
        "product_score_changed": False,
        "human_recording_gate_changed": False,
        "learner_quality_labels_inferred": False,
        "rows": rows,
        "summary": {
            "interpretation": "unlabeled_real_Japanese_L2_domain_sanity_not_pronunciation_validity",
            "public_minimal_pair": "じぶつ / じんぶつ",
            "special_mora_contrast": "N presence/absence",
            "ctc_peakiness_inventory_role": "all_acoustic_phone_events_including_special_morae",
            "required_for_validity": "full_criterion_labels_or_equivalent_expert_phone_labels",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    for row in pair_rows:
        diag = row["minimal_pair_diagnostic"]
        print(
            row["sample_id"],
            row["target_text"],
            "vs",
            diag["partner_text"],
            "target-minus-partner=",
            diag["page_target_minus_partner_sequence_logposterior"],
        )
    print("CTC POSTERIOR PHONE INVENTORY: ALL ACOUSTIC PHONE EVENTS INCLUDING N/cl")
    print("PUBLIC UME-JRF SAMPLE GRADES: NOT AVAILABLE / NO ERROR LABEL INFERENCE")
    print("HUMAN RECORDING GATE: UNCHANGED / BLOCKED")
    print("PRODUCT SCORE: UNCHANGED / RESEARCH ONLY")


if __name__ == "__main__":
    main()

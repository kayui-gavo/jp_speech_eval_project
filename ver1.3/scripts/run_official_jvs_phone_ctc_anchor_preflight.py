#!/usr/bin/env python3
"""Run three pinned Japanese phone-CTC backbones on official JVS samples.

This preflight uses the three human native-speaker sample clips linked directly
from the official JVS project page. It intentionally avoids phone-error claims:
there are no local pronunciation labels here. The only criterion is whether
the known target phone sequence is better supported than a deterministic
same-length phone-order permutation, plus stability to mild gain changes.

Models are loaded and released one at a time so a CPU CI runner never needs to
hold Beatrice, DistilHuBERT and WavLM simultaneously. Downloaded JVS audio is
ephemeral and must not be committed or uploaded as a workflow artifact.

The downloader's current manifest treats raw HTTP/WAV hashes as provenance, not
acoustic identity. This consumer therefore accepts the current ``raw_sha256``
field (and legacy ``sha256`` only for old saved artifacts) while preserving the
verified sample-rate/duration semantics in the derived report.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.audio_features import load_audio  # noqa: E402
from jp_speech_eval.dual_ctc_phone_candidate import (  # noqa: E402
    DISTILHUBERT_DUAL_CTC_MODEL,
    WAVLM_DUAL_CTC_MODEL,
    DualCtcPhoneCandidateBackend,
)
from jp_speech_eval.japanese_phoneme_gop import (  # noqa: E402
    DEFAULT_PHONE_CTC_REVISION,
    JapanesePhoneCtcBackend,
    ctc_forward_logprob,
    project_japanese_ctc_logits,
    sanitize_canonical_phones,
)
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


TARGET_TEXT = "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。"
DISTIL_REVISION = "01ffc3e5b0e49ba34180d50c48ea4111aa041cfd"
WAVLM_REVISION = "47fa985035342365bcec4948bd821aaf58dd778a"


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maxima = np.max(values, axis=1, keepdims=True)
    shifted = values - maxima
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _rotate_control(ids: Sequence[int]) -> list[int]:
    values = list(map(int, ids))
    if len(values) < 2:
        raise ValueError("phone permutation control requires at least two phones")
    shift = max(1, len(values) // 3)
    rotated = values[shift:] + values[:shift]
    if rotated == values:
        rotated = values[1:] + values[:1]
    return rotated


def _sequence_metrics(
    logits: np.ndarray,
    phones: Sequence[str],
    vocab: Dict[str, int],
    blank_id: int,
) -> Dict[str, Any]:
    missing = sorted({phone for phone in phones if phone not in vocab})
    if missing:
        return {"available": False, "reason": "phone_inventory_mismatch", "missing_phones": missing}
    log_probs = _log_softmax(logits)
    ids = [int(vocab[phone]) for phone in phones]
    control_ids = _rotate_control(ids)
    canonical_lp = ctc_forward_logprob(log_probs, ids, blank_id=int(blank_id))
    control_lp = ctc_forward_logprob(log_probs, control_ids, blank_id=int(blank_id))
    frames = int(logits.shape[0])
    if not math.isfinite(canonical_lp) or not math.isfinite(control_lp):
        return {"available": False, "reason": "nonfinite_sequence_logposterior"}
    return {
        "available": True,
        "frame_count": frames,
        "phone_count": len(phones),
        "canonical_log_posterior": float(canonical_lp),
        "permuted_same_length_log_posterior": float(control_lp),
        "canonical_minus_permuted": float(canonical_lp - control_lp),
        "canonical_log_posterior_per_frame": float(canonical_lp / frames),
        "permuted_log_posterior_per_frame": float(control_lp / frames),
        "canonical_minus_permuted_per_frame": float((canonical_lp - control_lp) / frames),
        "control": "deterministic_phone_rotation_same_phone_count",
        "control_is_pronunciation_error_label": False,
    }


def _source_provenance(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize current/legacy JVS manifest provenance without conflating hash with identity."""
    raw_hash = str(row.get("raw_sha256") or row.get("sha256") or "").strip()
    if not raw_hash:
        raise ValueError(f"JVS manifest row for {row.get('speaker')} is missing raw hash provenance")
    provenance: Dict[str, Any] = {
        "raw_sha256": raw_hash,
        "raw_transport_hash_is_acoustic_identity": bool(
            row.get("raw_transport_hash_is_acoustic_identity", False)
        ),
    }
    for key in (
        "raw_bytes",
        "bytes",
        "sample_rate",
        "sample_width_bytes",
        "frame_count",
        "duration_sec",
        "semantic_duration_verified",
        "raw_transport_variant_previously_observed",
        "google_drive_file_id",
    ):
        if key in row:
            provenance[key] = row[key]
    return provenance


class BeatriceInfer:
    name = "beatrice"
    model_id = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"
    revision = DEFAULT_PHONE_CTC_REVISION

    def __init__(self, *, allow_download: bool, device: str | None) -> None:
        self.backend = JapanesePhoneCtcBackend(
            device=device,
            local_files_only=not allow_download,
        )
        self.backend._load()

    def infer(self, waveform: np.ndarray) -> tuple[np.ndarray, Dict[str, int], int]:
        inputs = self.backend.processor(waveform, sampling_rate=16000, return_tensors="pt")
        model_inputs = {key: value.to(self.backend.device) for key, value in inputs.items()}
        with self.backend._torch.no_grad():
            output = self.backend.model(**model_inputs)
        raw_logits = output.logits.squeeze(0).detach().cpu().numpy()
        logical_logits, logical_vocab, _projection = project_japanese_ctc_logits(
            raw_logits, self.backend.vocabulary()
        )
        if "PAD" not in logical_vocab:
            raise RuntimeError("Beatrice logical blank token missing")
        return logical_logits, logical_vocab, int(logical_vocab["PAD"])


class DualInfer:
    def __init__(self, model_id: str, revision: str, *, allow_download: bool, device: str | None) -> None:
        self.model_id = model_id
        self.revision = revision
        self.name = "distilhubert_dual_ctc" if model_id == DISTILHUBERT_DUAL_CTC_MODEL else "wavlm_dual_ctc"
        self.backend = DualCtcPhoneCandidateBackend(
            model_id=model_id,
            revision=revision,
            device=device,
            local_files_only=not allow_download,
        )

    def infer(self, waveform: np.ndarray) -> tuple[np.ndarray, Dict[str, int], int]:
        logits, vocab, blank_id, _projection = self.backend.infer_logical_phone_logits(
            waveform, sr=16000
        )
        return logits, vocab, blank_id


def _load_manifest(path: Path) -> list[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("samples")
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("official JVS sample manifest is incomplete")
    return [dict(row) for row in rows]


def _run_model(model: Any, sample_rows: list[Dict[str, Any]], phones: list[str]) -> Dict[str, Any]:
    rows = []
    first_speech = None
    for row in sample_rows:
        audio = load_audio(str(Path(row["path"])), sr=16000)
        speech, region = trim_to_speech(audio.y, audio.sr)
        if first_speech is None:
            first_speech = np.asarray(speech, dtype=np.float32)
        logits, vocab, blank_id = model.infer(np.asarray(speech, dtype=np.float32))
        metrics = _sequence_metrics(logits, phones, vocab, blank_id)
        rows.append(
            {
                "speaker": row["speaker"],
                "source_provenance": _source_provenance(row),
                "speech_region": region.to_dict(),
                "metrics": metrics,
            }
        )

    gain_controls = []
    assert first_speech is not None
    clean_reference = rows[0]["metrics"]
    clean_pf = clean_reference.get("canonical_log_posterior_per_frame")
    for gain in (0.8, 1.2):
        logits, vocab, blank_id = model.infer(np.asarray(first_speech * gain, dtype=np.float32))
        metrics = _sequence_metrics(logits, phones, vocab, blank_id)
        current_pf = metrics.get("canonical_log_posterior_per_frame")
        gain_controls.append(
            {
                "gain": gain,
                "metrics": metrics,
                "canonical_per_frame_delta_from_clean": (
                    float(current_pf) - float(clean_pf)
                    if current_pf is not None and clean_pf is not None
                    else None
                ),
            }
        )

    margins = [
        float(row["metrics"]["canonical_minus_permuted_per_frame"])
        for row in rows
        if row["metrics"].get("available")
    ]
    canonical_pf = [
        float(row["metrics"]["canonical_log_posterior_per_frame"])
        for row in rows
        if row["metrics"].get("available")
    ]
    gain_abs_deltas = [
        abs(float(row["canonical_per_frame_delta_from_clean"]))
        for row in gain_controls
        if row.get("canonical_per_frame_delta_from_clean") is not None
    ]
    return {
        "name": model.name,
        "model_id": model.model_id,
        "revision": model.revision,
        "samples": rows,
        "gain_controls_jvs001": gain_controls,
        "summary": {
            "sample_count": len(rows),
            "all_samples_available": len(margins) == len(rows),
            "all_native_targets_beat_same_length_permutation": bool(margins) and all(value > 0 for value in margins),
            "canonical_minus_permuted_per_frame_min": min(margins) if margins else None,
            "canonical_minus_permuted_per_frame_mean": float(np.mean(margins)) if margins else None,
            "canonical_per_frame_cross_speaker_std": float(np.std(canonical_pf)) if canonical_pf else None,
            "max_abs_gain_delta_per_frame_jvs001": max(gain_abs_deltas) if gain_abs_deltas else None,
            "phone_correctness_claimed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/jvs_official_samples_manifest.json")
    parser.add_argument("--output", default="outputs/jvs_native_phone_ctc_anchor_preflight.json")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    target = build_japanese_target_evidence(TARGET_TEXT)
    phones, dropped = sanitize_canonical_phones(target.phones)
    sample_rows = _load_manifest(Path(args.manifest))
    for row in sample_rows:
        if str(row.get("target_text")) != TARGET_TEXT:
            raise ValueError(f"unexpected target text for {row.get('speaker')}")
        _source_provenance(row)

    factories: list[Callable[[], Any]] = [
        lambda: BeatriceInfer(allow_download=args.allow_download, device=args.device),
        lambda: DualInfer(
            DISTILHUBERT_DUAL_CTC_MODEL,
            DISTIL_REVISION,
            allow_download=args.allow_download,
            device=args.device,
        ),
        lambda: DualInfer(
            WAVLM_DUAL_CTC_MODEL,
            WAVLM_REVISION,
            allow_download=args.allow_download,
            device=args.device,
        ),
    ]
    results = []
    for factory in factories:
        model = factory()
        torch_module = getattr(getattr(model, "backend", None), "_torch", None)
        device = str(getattr(getattr(model, "backend", None), "device", "") or "")
        try:
            result = _run_model(model, sample_rows, phones)
            results.append(result)
            print(model.name, result["summary"])
        finally:
            del model
            gc.collect()
            if torch_module is not None and device.startswith("cuda") and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

    payload = {
        "schema": "jvs_native_phone_ctc_anchor_preflight_v2",
        "target_text": TARGET_TEXT,
        "canonical_phones": phones,
        "dropped_nonsegmental_target_tokens": dropped,
        "source": "official_JVS_project_page_three_sample_links",
        "source_identity_policy": "reviewed_file_id_plus_audio_semantics_raw_hash_provenance_only",
        "human_recordings_requested_from_user": False,
        "new_human_recordings_collected": False,
        "product_score_changed": False,
        "score_mapped": False,
        "criterion": "known_native_target_vs_same_length_phone_permutation_and_mild_gain_only",
        "local_phone_error_labels_available": False,
        "same_length_control_removes_phone_count_confound": True,
        "models": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print("PRODUCT SCORE: UNCHANGED / NATIVE ANCHOR PRECHECK ONLY")


if __name__ == "__main__":
    main()

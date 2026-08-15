#!/usr/bin/env python3
"""Run pinned Japanese phone-CTC backbones on official JVS native anchors.

The three official JVS sample WAVs are downloaded ephemerally by the workflow.
This preflight checks target-conditioned native behavior across the pinned
Beatrice, DistilHuBERT and WavLM phone-CTC backbones. It is an engineering/native
anchor, not learner-error validation.

A Stage-0 audit discovered that automatic surface-kanji G2P could read ``明王``
as ``あきらおう``. The anchor now *requires* the reviewed full-sentence kana
reading stored in the manifest and builds phones from that reading. Reintroducing
surface-only G2P for this anchor is treated as a provenance error.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Dict, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.audio_features import load_audio  # noqa: E402
from jp_speech_eval.ctc_sequence import ctc_forward_logprob_vectorized  # noqa: E402
from jp_speech_eval.dual_ctc_phone_candidate import (  # noqa: E402
    DISTILHUBERT_DUAL_CTC_MODEL,
    WAVLM_DUAL_CTC_MODEL,
    DualCtcPhoneCandidateBackend,
)
from jp_speech_eval.japanese_phoneme_gop import (  # noqa: E402
    DEFAULT_PHONE_CTC_MODEL,
    DEFAULT_PHONE_CTC_REVISION,
    JapanesePhoneCtcBackend,
    project_japanese_ctc_logits,
    sanitize_canonical_phones,
)
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


TARGET_TEXT = "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。"
DISTIL_REVISION = "01ffc3e5b0e49ba34180d50c48ea4111aa041cfd"
WAVLM_REVISION = "47fa985035342365bcec4948bd821aaf58dd778a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="outputs/jvs_native_phone_ctc_anchor_preflight.json")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


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


def _sequence_metrics(logits: np.ndarray, phones: Sequence[str], vocab: Dict[str, int], blank_id: int) -> Dict[str, Any]:
    missing = sorted({str(phone) for phone in phones if str(phone) not in vocab})
    if missing:
        return {"available": False, "reason": "phone_inventory_mismatch", "missing_phones": missing}
    token_ids = [int(vocab[str(phone)]) for phone in phones]
    if not token_ids:
        return {"available": False, "reason": "empty_phone_sequence"}
    log_probs = _log_softmax(logits)
    canonical = ctc_forward_logprob_vectorized(log_probs, token_ids, blank_id=int(blank_id))
    permuted_ids = _rotate_control(token_ids)
    permuted = ctc_forward_logprob_vectorized(log_probs, permuted_ids, blank_id=int(blank_id))
    frame_count = int(log_probs.shape[0])
    return {
        "available": bool(np.isfinite(canonical) and np.isfinite(permuted)),
        "canonical_logposterior": float(canonical),
        "permuted_control_logposterior": float(permuted),
        "canonical_minus_permuted": float(canonical - permuted),
        "canonical_minus_permuted_per_frame": float((canonical - permuted) / frame_count),
        "canonical_logposterior_per_frame": float(canonical / frame_count),
        "permuted_control_logposterior_per_frame": float(permuted / frame_count),
        "frame_count": frame_count,
        "phone_count": len(token_ids),
        "control": "deterministic_phone_rotation_same_inventory_and_length",
        "control_is_pronunciation_error_label": False,
        "interpretation": "target_specificity_engineering_anchor_not_pronunciation_validity",
    }


def _source_provenance(sample: Dict[str, Any]) -> Dict[str, Any]:
    raw_hash = str(sample.get("raw_sha256") or sample.get("sha256") or "")
    if not raw_hash:
        raise ValueError(f"JVS sample {sample.get('speaker')} missing raw hash provenance")
    provenance = {
        "speaker": sample.get("speaker"),
        "raw_sha256": raw_hash,
        "raw_transport_hash_is_acoustic_identity": bool(sample.get("raw_transport_hash_is_acoustic_identity", False)),
        "google_drive_file_id": sample.get("google_drive_file_id"),
        "sample_rate": sample.get("sample_rate"),
        "sample_width_bytes": sample.get("sample_width_bytes"),
        "duration_sec": sample.get("duration_sec"),
        "semantic_duration_verified": bool(sample.get("semantic_duration_verified", False)),
        "target_reading": sample.get("target_reading"),
        "target_reading_source": sample.get("target_reading_source"),
        "automatic_surface_g2p_is_safe_for_anchor": sample.get("automatic_surface_g2p_is_safe_for_anchor"),
    }
    if "bytes" in sample:
        provenance["bytes"] = sample.get("bytes")
    if "raw_bytes" in sample:
        provenance["raw_bytes"] = sample.get("raw_bytes")
    return provenance


def _load_manifest(path: Path) -> list[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = str(payload.get("schema") or "")
    if schema not in {"jvs_official_samples_manifest_v2", "jvs_official_samples_manifest_v3", "jvs_official_samples_manifest_v4"}:
        raise ValueError(f"unexpected JVS manifest schema: {schema!r}")
    rows = list(payload.get("samples") or [])
    if {str(row.get("speaker")) for row in rows} != {"jvs001", "jvs002", "jvs003"}:
        raise ValueError("JVS manifest must contain exactly jvs001/jvs002/jvs003")
    if any(str(row.get("target_text") or "") != TARGET_TEXT for row in rows):
        raise ValueError("JVS manifest target text drift")
    return rows


def _reviewed_reading(rows: Sequence[Dict[str, Any]]) -> str:
    readings = {str(row.get("target_reading") or "").strip() for row in rows}
    if "" in readings or len(readings) != 1:
        raise ValueError("JVS manifest requires one reviewed target_reading for all speakers")
    if any(bool(row.get("automatic_surface_g2p_is_safe_for_anchor", True)) for row in rows):
        raise ValueError("JVS anchor must explicitly forbid automatic surface-only G2P")
    return readings.pop()


class BeatriceInfer:
    def __init__(self, *, allow_download: bool, device: str | None) -> None:
        self.backend = JapanesePhoneCtcBackend(
            model_id=DEFAULT_PHONE_CTC_MODEL,
            revision=DEFAULT_PHONE_CTC_REVISION,
            device=device,
            local_files_only=not allow_download,
        )
        self.model_id = DEFAULT_PHONE_CTC_MODEL
        self.revision = DEFAULT_PHONE_CTC_REVISION

    def infer(self, waveform: np.ndarray):
        backend = self.backend
        backend._load()
        inputs = backend.processor(np.asarray(waveform, dtype=np.float32), sampling_rate=16000, return_tensors="pt")
        model_inputs = {key: value.to(backend.device) for key, value in inputs.items()}
        with backend._torch.no_grad():
            output = backend.model(**model_inputs)
        raw_logits = output.logits.squeeze(0).detach().cpu().numpy()
        logical_logits, logical_vocab, _projection = project_japanese_ctc_logits(raw_logits, backend.vocabulary())
        return logical_logits, logical_vocab, int(logical_vocab["PAD"])

    def close(self) -> None:
        self.backend.processor = None
        self.backend.model = None
        self.backend.tokenizer = None
        gc.collect()


class DualInfer:
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
        logits, vocab, blank_id, _provenance = self.backend.infer_logical_phone_logits(np.asarray(waveform, dtype=np.float32), sr=16000)
        return logits, vocab, int(blank_id)

    def close(self) -> None:
        self.backend.processor = None
        self.backend.model = None
        self.backend.tokenizer = None
        gc.collect()


def _model_specs():
    return [
        ("beatrice", DEFAULT_PHONE_CTC_MODEL, DEFAULT_PHONE_CTC_REVISION),
        ("distilhubert_dual_ctc", DISTILHUBERT_DUAL_CTC_MODEL, DISTIL_REVISION),
        ("wavlm_dual_ctc", WAVLM_DUAL_CTC_MODEL, WAVLM_REVISION),
    ]


def _build_model(key: str, model_id: str, revision: str, *, allow_download: bool, device: str | None):
    if key == "beatrice":
        return BeatriceInfer(allow_download=allow_download, device=device)
    return DualInfer(model_id, revision, allow_download=allow_download, device=device)


def main() -> None:
    args = parse_args()
    rows = _load_manifest(Path(args.manifest))
    reading = _reviewed_reading(rows)
    target = build_japanese_target_evidence(TARGET_TEXT, reading_override=reading)
    phones, dropped = sanitize_canonical_phones(target.phones)

    loaded_samples = []
    for sample in rows:
        audio = load_audio(str(Path(sample["path"])), sr=16000)
        speech, region = trim_to_speech(audio.y, audio.sr)
        loaded_samples.append((sample, np.asarray(speech, dtype=np.float32), region.to_dict()))

    results_by_speaker: Dict[str, Dict[str, Any]] = {
        str(sample["speaker"]): {
            "speaker": sample["speaker"],
            "source_provenance": _source_provenance(sample),
            "speech_region": region,
            "models": [],
        }
        for sample, _speech, region in loaded_samples
    }

    for key, model_id, revision in _model_specs():
        model = _build_model(key, model_id, revision, allow_download=bool(args.allow_download), device=args.device)
        try:
            for sample, speech, _region in loaded_samples:
                try:
                    logits, vocab, blank_id = model.infer(speech)
                    metrics = _sequence_metrics(logits, phones, vocab, blank_id)
                except Exception as exc:
                    metrics = {
                        "available": False,
                        "reason": "model_inference_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                results_by_speaker[str(sample["speaker"])]["models"].append(
                    {"model_key": key, "model_id": model_id, "revision": revision, "metrics": metrics}
                )
        finally:
            model.close()

    samples = [results_by_speaker[str(row["speaker"])] for row in rows]
    payload = {
        "schema": "jvs_native_phone_ctc_anchor_preflight_v3",
        "source": "official_JVS_public_sample_links_ephemeral",
        "target_text": TARGET_TEXT,
        "target_reading": reading,
        "target_reading_source": "reviewed_manifest_override",
        "automatic_surface_g2p_used": False,
        "known_surface_g2p_failure_fixed": "明王:auto=あきらおう,target=みょうおう",
        "target_phones": phones,
        "dropped_nonsegmental_target_tokens": dropped,
        "native_audio_has_phone_error_labels": False,
        "control_is_pronunciation_error_label": False,
        "score_mapped": False,
        "product_calibrated": False,
        "product_score_changed": False,
        "human_recording_gate_changed": False,
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print("target reading override:", reading)
    for sample in samples:
        for model in sample["models"]:
            metrics = model["metrics"]
            print(sample["speaker"], model["model_key"], "available=", metrics.get("available"), "gap/frame=", metrics.get("canonical_minus_permuted_per_frame"))
    if any(not bool(model["metrics"].get("available")) for sample in samples for model in sample["models"]):
        raise SystemExit("one or more JVS native model anchors unavailable")
    print("HUMAN RECORDING GATE: UNCHANGED / BLOCKED")
    print("PRODUCT SCORE: UNCHANGED / RESEARCH ONLY")


if __name__ == "__main__":
    main()

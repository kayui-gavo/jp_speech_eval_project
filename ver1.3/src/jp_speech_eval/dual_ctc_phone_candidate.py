"""Adapter for alternative Japanese dual-CTC phone models in Stage-0 research.

The adapter is intentionally shadow-only. A caller must provide an immutable
Hugging Face revision; mutable ``main`` is rejected. The model's remote custom
code is only trusted at that pinned revision. Raw phoneme logits are projected
into the same logical Japanese phone space used by the Beatrice research
backend so candidate comparisons can use identical CTC evidence functions.

No learner-facing score is produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .japanese_phoneme_gop import (
    project_japanese_ctc_logits,
    sanitize_canonical_phones,
    segmental_competitor_ids,
)
from .phoneme_gop import PhoneGopResult, compute_phone_gop_evidence
from .segmentation_free_gop import SegmentationFreeGopResult, compute_enumerated_fgop_sf_sd_features


DISTILHUBERT_DUAL_CTC_MODEL = "TylorShine/distilhubert-hiragana-ctc"
WAVLM_DUAL_CTC_MODEL = "TylorShine/wavlm-base-plus-hiragana-ctc"


def validate_immutable_revision(revision: str) -> str:
    value = str(revision or "").strip()
    if not value or value in {"main", "master", "latest"}:
        raise ValueError("dual-CTC candidate requires an immutable Hugging Face revision SHA")
    if len(value) < 12:
        raise ValueError("dual-CTC candidate revision looks too short to be an immutable commit SHA")
    return value


def normalize_phoneme_vocab(vocab: Mapping[str, int]) -> Tuple[Dict[str, int], int]:
    """Validate the candidate phoneme tokenizer and return raw vocab + blank id."""
    normalized = {str(token): int(index) for token, index in vocab.items()}
    blank_names = [name for name in ("<blank>", "PAD", "<pad>") if name in normalized]
    if len(blank_names) != 1:
        raise ValueError(f"expected exactly one CTC blank token, found {blank_names}")
    blank_id = int(normalized[blank_names[0]])
    if blank_id < 0:
        raise ValueError("CTC blank token id must be non-negative")
    return normalized, blank_id


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_phoneme_logits(output: Any) -> np.ndarray:
    """Extract a ``(frames, phone_vocab)`` array from the custom model output."""
    value = None
    if isinstance(output, Mapping):
        value = output.get("phoneme_logits")
    if value is None:
        value = getattr(output, "phoneme_logits", None)
    if value is None:
        raise ValueError("dual-CTC model output does not expose phoneme_logits")
    array = _tensor_to_numpy(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 1:
        raise ValueError(f"unexpected phoneme_logits shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("dual-CTC phoneme logits contain non-finite values")
    return np.asarray(array, dtype=np.float64)


@dataclass
class DualCtcPhoneCandidateBackend:
    model_id: str
    revision: str
    device: Optional[str] = None
    local_files_only: bool = True

    def __post_init__(self) -> None:
        self.revision = validate_immutable_revision(self.revision)
        self.model = None
        self.feature_extractor = None
        self.phoneme_tokenizer = None
        self._torch = None
        self._raw_vocab: Dict[str, int] = {}
        self._raw_blank_id: Optional[int] = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel, PreTrainedTokenizerFast
        except ImportError as exc:
            raise RuntimeError("dual-CTC candidate requires torch and transformers") from exc

        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        common = {
            "revision": self.revision,
            "local_files_only": self.local_files_only,
        }
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id, **common)
        self.phoneme_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.model_id,
            subfolder="phoneme_tokenizer",
            **common,
        )
        # Remote custom code is accepted only because revision is immutable.
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **common,
        )
        self.model.to(self.device)
        self.model.eval()
        self._raw_vocab, self._raw_blank_id = normalize_phoneme_vocab(self.phoneme_tokenizer.get_vocab())
        self._validate_contract()

    def _expected_sample_rate(self) -> int:
        value = getattr(self.feature_extractor, "sampling_rate", None) if self.feature_extractor is not None else None
        return int(value or 16000)

    def _validate_contract(self) -> None:
        if self._expected_sample_rate() != 16000:
            raise RuntimeError("dual-CTC candidate is expected to use 16 kHz audio")
        if self._raw_blank_id is None:
            raise RuntimeError("dual-CTC candidate blank id was not initialized")
        config_size = int(getattr(self.model.config, "phoneme_vocab_size", 0) or 0)
        if config_size and max(self._raw_vocab.values(), default=-1) >= config_size:
            raise RuntimeError("phoneme tokenizer vocabulary exceeds candidate phoneme head size")
        for required in ("a", "i", "u", "N", "cl"):
            if required not in self._raw_vocab and required.upper() not in self._raw_vocab:
                raise RuntimeError(f"required Japanese phone missing from candidate vocabulary: {required}")

    def infer_logical_phone_logits(
        self,
        audio: np.ndarray,
        *,
        sr: int = 16000,
    ) -> Tuple[np.ndarray, Dict[str, int], int, Dict[str, Sequence[str]]]:
        self._load()
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise ValueError("empty_audio")
        if not np.isfinite(waveform).all():
            raise ValueError("nonfinite_audio")
        if int(sr) != self._expected_sample_rate():
            raise ValueError("sampling_rate_mismatch")
        if waveform.size < int(0.12 * sr):
            raise ValueError("audio_too_short")
        if float(np.max(np.abs(waveform))) < 1e-5:
            raise ValueError("near_silent_audio")

        inputs = self.feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")
        model_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            output = self.model(**model_inputs)
        raw_logits = extract_phoneme_logits(output)
        if raw_logits.shape[1] <= max(self._raw_vocab.values(), default=-1):
            raise ValueError("candidate phoneme logits are smaller than tokenizer vocabulary")
        logical_logits, logical_vocab, provenance = project_japanese_ctc_logits(raw_logits, self._raw_vocab)
        raw_blank_name = next(name for name in ("<blank>", "PAD", "<pad>") if name in self._raw_vocab)
        if raw_blank_name not in logical_vocab:
            raise ValueError("logical blank token missing after projection")
        return logical_logits, logical_vocab, int(logical_vocab[raw_blank_name]), provenance

    def evaluate_frame_local(
        self,
        audio: np.ndarray,
        canonical_phones: Sequence[str],
        *,
        sr: int = 16000,
    ) -> PhoneGopResult:
        phones, dropped = sanitize_canonical_phones(canonical_phones)
        try:
            logits, vocab, blank_id, provenance = self.infer_logical_phone_logits(audio, sr=sr)
        except Exception as exc:
            return PhoneGopResult(
                available=False,
                backend="hf_dual_ctc_phone_candidate",
                model_id=f"{self.model_id}@{self.revision}",
                method="dual_ctc_frame_local_diagnostic_v1",
                canonical_phones=phones,
                evidence=[],
                summary={"reason": f"candidate_inference_failed:{type(exc).__name__}", "detail": str(exc)},
                warnings=["candidate_inference_failed"],
            )
        missing = sorted({phone for phone in phones if phone not in vocab})
        if missing:
            return PhoneGopResult(
                available=False,
                backend="hf_dual_ctc_phone_candidate",
                model_id=f"{self.model_id}@{self.revision}",
                method="dual_ctc_frame_local_diagnostic_v1",
                canonical_phones=phones,
                evidence=[],
                summary={"reason": "phone_inventory_mismatch", "missing_phones": missing},
                warnings=["phone_inventory_mismatch"],
            )
        competitor_ids = segmental_competitor_ids(vocab, blank_id=blank_id)
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        frame_stride_sec = float(waveform.size) / float(sr) / float(logits.shape[0])
        result = compute_phone_gop_evidence(
            logits,
            phones,
            vocab=vocab,
            blank_id=blank_id,
            frame_stride_sec=frame_stride_sec,
            backend="hf_dual_ctc_phone_candidate",
            model_id=f"{self.model_id}@{self.revision}",
            competitor_token_ids=competitor_ids,
        )
        summary = dict(result.summary)
        summary.update(
            {
                "candidate_model_id": self.model_id,
                "candidate_revision": self.revision,
                "remote_custom_code_pinned": True,
                "logical_phone_projection": provenance,
                "dropped_nonsegmental_target_tokens": dropped,
                "frame_stride_sec_observed_average": frame_stride_sec,
                "ctc_support_frames_are_not_physical_phone_boundaries": True,
            }
        )
        return PhoneGopResult(
            available=result.available,
            backend=result.backend,
            model_id=result.model_id,
            method="dual_ctc_frame_local_diagnostic_v1",
            canonical_phones=result.canonical_phones,
            evidence=result.evidence,
            summary=summary,
            warnings=result.warnings,
            score_mapped=False,
            product_calibrated=False,
        )

    def evaluate_segmentation_free(
        self,
        audio: np.ndarray,
        canonical_phones: Sequence[str],
        *,
        sr: int = 16000,
    ) -> SegmentationFreeGopResult:
        phones, _dropped = sanitize_canonical_phones(canonical_phones)
        logits, vocab, blank_id, _provenance = self.infer_logical_phone_logits(audio, sr=sr)
        return compute_enumerated_fgop_sf_sd_features(
            logits,
            phones,
            vocab=vocab,
            blank_id=blank_id,
            model_id=self.model_id,
            revision=self.revision,
        )

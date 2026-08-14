"""Japanese phone-level GOP evidence for research/shadow use.

This module intentionally separates *evidence extraction* from product score
mapping.  It provides:

1. a pure NumPy CTC Viterbi alignment over a known canonical phone sequence;
2. phone-local posterior/logit/uncertainty features inspired by GOP research;
3. a lazy Hugging Face phone-CTC backend, defaulting to a Japanese HuBERT
   phoneme model when explicitly enabled.

Nothing here creates a learner-facing /100 score.  In particular, native ASR
phone recognition accuracy is not treated as proof of L2 pronunciation-score
validity.  Ordinary imports and unit tests make no network requests and do not
load model weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


DEFAULT_PHONE_CTC_MODEL = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"


@dataclass(frozen=True)
class PhoneGopEvidence:
    phone_index: int
    canonical_phone: str
    token_id: int
    start_frame: int
    end_frame: int
    frame_count: int
    start_sec: float
    end_sec: float
    duration_sec: float
    target_mean_logit: float
    target_max_logit: float
    target_mean_logprob: float
    best_competitor_phone: str
    best_competitor_token_id: int
    best_competitor_mean_logit: float
    best_competitor_max_logit: float
    best_competitor_mean_logprob: float
    mean_logit_margin: float
    max_logit_margin: float
    posterior_gop_margin: float
    mean_entropy: float
    path_support_mean_logprob: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhoneGopResult:
    available: bool
    backend: str
    model_id: str
    method: str
    canonical_phones: List[str]
    evidence: List[PhoneGopEvidence]
    summary: Dict[str, Any]
    warnings: List[str]
    score_mapped: bool = False
    product_calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = [item.to_dict() for item in self.evidence]
        return payload


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("CTC logits must have shape (frames, vocabulary)")
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("CTC logits must contain frames and at least two tokens")
    maxima = np.max(values, axis=1, keepdims=True)
    shifted = values - maxima
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _extended_ctc_labels(token_ids: Sequence[int], blank_id: int) -> List[int]:
    if not token_ids:
        return [int(blank_id)]
    out: List[int] = [int(blank_id)]
    for token_id in token_ids:
        out.extend([int(token_id), int(blank_id)])
    return out


def ctc_viterbi_align(
    log_probs: np.ndarray,
    token_ids: Sequence[int],
    *,
    blank_id: int,
) -> Dict[str, Any]:
    """Viterbi-align a known token sequence under the standard CTC topology.

    The returned phone states are CTC *support frames*, not conventional HMM
    phone segments.  CTC outputs can be very peaky; callers must preserve that
    distinction and should not interpret one-frame support as a true duration.
    This implementation exists so different GOP-style feature definitions can
    be compared without an external forced aligner.
    """
    probs = np.asarray(log_probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("log_probs must have shape (frames, vocabulary)")
    frames, vocab_size = probs.shape
    if frames <= 0:
        raise ValueError("CTC alignment requires at least one frame")
    tokens = [int(value) for value in token_ids]
    if any(value < 0 or value >= vocab_size for value in tokens):
        raise ValueError("canonical token id is outside the CTC vocabulary")
    if blank_id < 0 or blank_id >= vocab_size:
        raise ValueError("blank_id is outside the CTC vocabulary")
    if not tokens:
        return {
            "state_path": [0] * frames,
            "extended_labels": [int(blank_id)],
            "phone_frame_indices": [],
            "path_logprob": float(np.sum(probs[:, blank_id])),
        }

    labels = _extended_ctc_labels(tokens, blank_id)
    states = len(labels)
    neg_inf = -np.inf
    dp = np.full((frames, states), neg_inf, dtype=np.float64)
    back = np.full((frames, states), -1, dtype=np.int32)

    dp[0, 0] = probs[0, blank_id]
    if states > 1:
        dp[0, 1] = probs[0, labels[1]]

    for t in range(1, frames):
        # At frame t, no path can have advanced farther than roughly 2t+1
        # states; limiting the loop is a small but useful optimization.
        max_state = min(states, 2 * t + 2)
        for s in range(max_state):
            candidates: List[tuple[float, int]] = [(dp[t - 1, s], s)]
            if s - 1 >= 0:
                candidates.append((dp[t - 1, s - 1], s - 1))
            if s - 2 >= 0 and labels[s] != blank_id and labels[s] != labels[s - 2]:
                candidates.append((dp[t - 1, s - 2], s - 2))
            best_value, best_state = max(candidates, key=lambda item: item[0])
            if np.isfinite(best_value):
                dp[t, s] = best_value + probs[t, labels[s]]
                back[t, s] = best_state

    end_candidates = [states - 1]
    if states > 1:
        end_candidates.append(states - 2)
    final_state = max(end_candidates, key=lambda s: dp[-1, s])
    if not np.isfinite(dp[-1, final_state]):
        raise ValueError(
            "canonical phone sequence cannot be aligned to the available CTC frames"
        )

    state_path = [int(final_state)]
    for t in range(frames - 1, 0, -1):
        previous = int(back[t, state_path[-1]])
        if previous < 0:
            raise ValueError("CTC Viterbi backtrace is incomplete")
        state_path.append(previous)
    state_path.reverse()

    phone_frame_indices: List[List[int]] = []
    for phone_index in range(len(tokens)):
        phone_state = 2 * phone_index + 1
        phone_frame_indices.append(
            [frame for frame, state in enumerate(state_path) if state == phone_state]
        )

    return {
        "state_path": state_path,
        "extended_labels": labels,
        "phone_frame_indices": phone_frame_indices,
        "path_logprob": float(dp[-1, final_state]),
    }


def _id_to_token_map(vocab: Mapping[str, int]) -> Dict[int, str]:
    return {int(index): str(token) for token, index in vocab.items()}


def _finite_summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if not arr.size:
        return {"mean": None, "median": None, "q25": None, "min": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "min": float(np.min(arr)),
    }


def compute_phone_gop_evidence(
    logits: np.ndarray,
    canonical_phones: Sequence[str],
    *,
    vocab: Mapping[str, int],
    blank_id: int,
    frame_stride_sec: float,
    backend: str = "ctc_phone_model",
    model_id: str = "",
    competitor_token_ids: Optional[Sequence[int]] = None,
) -> PhoneGopResult:
    """Compute raw phone-local GOP/logit evidence without score calibration.

    This is deliberately a *feature extractor*.  The `posterior_gop_margin`
    resembles classic phone-competition GOP, while the logit margins are kept
    separately because recent work has shown that softmax GOP can be
    overconfident and that logit-based variants can behave differently.
    """
    raw_logits = np.asarray(logits, dtype=np.float64)
    if raw_logits.ndim != 2:
        raise ValueError("logits must have shape (frames, vocabulary)")
    if frame_stride_sec <= 0:
        raise ValueError("frame_stride_sec must be positive")

    phones = [str(phone) for phone in canonical_phones if str(phone)]
    warnings: List[str] = []
    missing = [phone for phone in phones if phone not in vocab]
    if missing:
        return PhoneGopResult(
            available=False,
            backend=backend,
            model_id=model_id,
            method="ctc_viterbi_phone_evidence_v1",
            canonical_phones=phones,
            evidence=[],
            summary={
                "reason": "canonical_phone_not_in_backend_vocabulary",
                "missing_phones": sorted(set(missing)),
            },
            warnings=["phone_inventory_mismatch"],
        )
    if not phones:
        return PhoneGopResult(
            available=False,
            backend=backend,
            model_id=model_id,
            method="ctc_viterbi_phone_evidence_v1",
            canonical_phones=[],
            evidence=[],
            summary={"reason": "empty_canonical_phone_sequence"},
            warnings=["empty_phone_sequence"],
        )

    token_ids = [int(vocab[phone]) for phone in phones]
    log_probs = _log_softmax(raw_logits)
    alignment = ctc_viterbi_align(log_probs, token_ids, blank_id=int(blank_id))

    id_to_token = _id_to_token_map(vocab)
    vocab_size = raw_logits.shape[1]
    if competitor_token_ids is None:
        special_ids = {int(blank_id)}
        for name in ("PAD", "UNK", "SOS", "EOS", "<pad>", "<unk>", "<s>", "</s>"):
            if name in vocab:
                special_ids.add(int(vocab[name]))
        competitor_ids = [index for index in range(vocab_size) if index not in special_ids]
    else:
        competitor_ids = [
            int(index) for index in competitor_token_ids
            if 0 <= int(index) < vocab_size and int(index) != int(blank_id)
        ]
    if not competitor_ids:
        raise ValueError("no competitor phone tokens are available")

    evidence: List[PhoneGopEvidence] = []
    support_counts: List[int] = []
    for phone_index, (phone, target_id, frames) in enumerate(
        zip(phones, token_ids, alignment["phone_frame_indices"])
    ):
        if not frames:
            # Standard CTC Viterbi normally emits every canonical label at least
            # once, but keep an explicit failure rather than synthesizing data.
            warnings.append(f"phone_without_ctc_support:{phone_index}:{phone}")
            continue
        frame_idx = np.asarray(frames, dtype=int)
        support_counts.append(int(len(frame_idx)))
        phone_logits = raw_logits[frame_idx]
        phone_log_probs = log_probs[frame_idx]

        target_logits = phone_logits[:, target_id]
        target_log_probs = phone_log_probs[:, target_id]

        competitor_candidates = [index for index in competitor_ids if index != target_id]
        if not competitor_candidates:
            competitor_candidates = [index for index in range(vocab_size) if index not in {target_id, blank_id}]
        competitor_matrix_logits = phone_logits[:, competitor_candidates]
        competitor_matrix_log_probs = phone_log_probs[:, competitor_candidates]
        competitor_mean_logits = np.mean(competitor_matrix_logits, axis=0)
        best_position = int(np.argmax(competitor_mean_logits))
        best_id = int(competitor_candidates[best_position])
        best_mean_logit = float(competitor_mean_logits[best_position])
        best_mean_logprob = float(np.mean(phone_log_probs[:, best_id]))

        competitor_max_logits = np.max(competitor_matrix_logits, axis=0)
        best_max_position = int(np.argmax(competitor_max_logits))
        best_max_id = int(competitor_candidates[best_max_position])
        best_max_logit = float(competitor_max_logits[best_max_position])

        probs = np.exp(phone_log_probs)
        entropy = -np.sum(probs * phone_log_probs, axis=1)
        start_frame = int(frame_idx[0])
        end_frame = int(frame_idx[-1] + 1)

        evidence.append(
            PhoneGopEvidence(
                phone_index=phone_index,
                canonical_phone=phone,
                token_id=target_id,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_count=int(len(frame_idx)),
                start_sec=float(start_frame * frame_stride_sec),
                end_sec=float(end_frame * frame_stride_sec),
                duration_sec=float(len(frame_idx) * frame_stride_sec),
                target_mean_logit=float(np.mean(target_logits)),
                target_max_logit=float(np.max(target_logits)),
                target_mean_logprob=float(np.mean(target_log_probs)),
                best_competitor_phone=id_to_token.get(best_id, str(best_id)),
                best_competitor_token_id=best_id,
                best_competitor_mean_logit=best_mean_logit,
                best_competitor_max_logit=best_max_logit,
                best_competitor_mean_logprob=best_mean_logprob,
                mean_logit_margin=float(np.mean(target_logits) - best_mean_logit),
                max_logit_margin=float(np.max(target_logits) - best_max_logit),
                posterior_gop_margin=float(np.mean(target_log_probs) - best_mean_logprob),
                mean_entropy=float(np.mean(entropy)),
                path_support_mean_logprob=float(np.mean(target_log_probs)),
            )
        )

    if len(evidence) != len(phones):
        return PhoneGopResult(
            available=False,
            backend=backend,
            model_id=model_id,
            method="ctc_viterbi_phone_evidence_v1",
            canonical_phones=phones,
            evidence=evidence,
            summary={
                "reason": "incomplete_ctc_phone_support",
                "canonical_phone_count": len(phones),
                "supported_phone_count": len(evidence),
            },
            warnings=warnings,
        )

    posterior_summary = _finite_summary(item.posterior_gop_margin for item in evidence)
    logit_summary = _finite_summary(item.mean_logit_margin for item in evidence)
    entropy_summary = _finite_summary(item.mean_entropy for item in evidence)
    sparse_ratio = float(np.mean([count <= 1 for count in support_counts])) if support_counts else 1.0
    if sparse_ratio >= 0.50:
        warnings.append("ctc_support_is_peaky_for_at_least_half_of_phones")

    return PhoneGopResult(
        available=True,
        backend=backend,
        model_id=model_id,
        method="ctc_viterbi_phone_evidence_v1",
        canonical_phones=phones,
        evidence=evidence,
        summary={
            "canonical_phone_count": len(phones),
            "supported_phone_count": len(evidence),
            "frame_stride_sec": float(frame_stride_sec),
            "ctc_path_logprob": float(alignment["path_logprob"]),
            "single_frame_support_ratio": sparse_ratio,
            "posterior_gop_margin": posterior_summary,
            "mean_logit_margin": logit_summary,
            "entropy": entropy_summary,
            "weakest_phone_by_posterior_margin": min(
                evidence, key=lambda item: item.posterior_gop_margin
            ).canonical_phone,
            "weakest_phone_index_by_posterior_margin": min(
                evidence, key=lambda item: item.posterior_gop_margin
            ).phone_index,
            "interpretation": (
                "raw_phone_competition_evidence_not_pronunciation_correctness_or_percent_score"
            ),
        },
        warnings=warnings,
    )


class HuggingFacePhoneCtcBackend:
    """Lazy Japanese phone-CTC backend for explicit shadow experiments.

    `local_files_only=True` is the safe default so ordinary product/test runs do
    not unexpectedly download hundreds of megabytes.  A benchmark can opt in to
    a download by constructing the backend with `local_files_only=False`.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_PHONE_CTC_MODEL,
        *,
        device: Optional[str] = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.processor = None
        self.model = None
        self._torch = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("phone-CTC GOP shadow requires torch and transformers") from exc
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
            )
            self.model = AutoModelForCTC.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to load phone-CTC model {self.model_id}: {exc}") from exc
        self.model.to(self.device)
        self.model.eval()

    def vocabulary(self) -> Dict[str, int]:
        self._load()
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "get_vocab"):
            raise RuntimeError("phone-CTC processor does not expose a tokenizer vocabulary")
        return {str(token): int(index) for token, index in tokenizer.get_vocab().items()}

    def _frame_stride_sec(self, audio_length: int, sr: int, frame_count: int) -> float:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        # Prefer the model's convolutional output-length calculation because it
        # remains valid when the exact frontend stride changes.  Time is then
        # distributed over the analyzed waveform; this is sufficient for CTC
        # support diagnostics and avoids pretending to have gold boundaries.
        duration = float(audio_length) / max(int(sr), 1)
        return duration / frame_count

    def evaluate(
        self,
        audio: np.ndarray,
        canonical_phones: Sequence[str],
        *,
        sr: int = 16000,
    ) -> PhoneGopResult:
        self._load()
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            return PhoneGopResult(
                available=False,
                backend="huggingface_phone_ctc",
                model_id=self.model_id,
                method="ctc_viterbi_phone_evidence_v1",
                canonical_phones=list(canonical_phones),
                evidence=[],
                summary={"reason": "empty_audio"},
                warnings=["empty_audio"],
            )
        inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt")
        model_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            output = self.model(**model_inputs)
        logits = output.logits.squeeze(0).detach().cpu().numpy()
        vocab = self.vocabulary()
        blank_id = int(getattr(self.model.config, "pad_token_id", 0) or 0)
        frame_stride_sec = self._frame_stride_sec(len(waveform), sr, logits.shape[0])
        return compute_phone_gop_evidence(
            logits,
            canonical_phones,
            vocab=vocab,
            blank_id=blank_id,
            frame_stride_sec=frame_stride_sec,
            backend="huggingface_phone_ctc",
            model_id=self.model_id,
        )


def evaluate_text_phone_gop_shadow(
    audio: np.ndarray,
    text: str,
    *,
    sr: int = 16000,
    backend: Optional[HuggingFacePhoneCtcBackend] = None,
) -> PhoneGopResult:
    """Convenience wrapper: Japanese text -> pyopenjtalk phones -> CTC evidence."""
    from .japanese_target_evidence import build_japanese_target_evidence

    target = build_japanese_target_evidence(text)
    runner = backend or HuggingFacePhoneCtcBackend()
    return runner.evaluate(audio, target.phones, sr=sr)

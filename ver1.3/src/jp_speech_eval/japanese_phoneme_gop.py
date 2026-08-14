"""Japanese-specific hardening for phone-CTC GOP shadow experiments.

This module sits on top of the generic ``phoneme_gop`` feature extractor and
handles Japanese/backend details that must be resolved before human recording:

* collapse legitimate high-vowel devoicing labels (i/I, u/U) into logical
  allophone classes for clarity scoring;
* keep pau/sil out of segmental competitor sets;
* pin the default Beatrice v4 model revision for reproducibility;
* validate model/tokenizer/sample-rate compatibility;
* expose sequence-level CTC evidence, including leave-one-phone-out deletion
  preference and greedy edit operations, so a forced canonical Viterbi path is
  not the only signal available for deletion/insertion stress tests.

Nothing in this module maps raw evidence to a learner-facing /100 score.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .phoneme_gop import (
    DEFAULT_PHONE_CTC_MODEL,
    HuggingFacePhoneCtcBackend,
    PhoneGopResult,
    compute_phone_gop_evidence,
)


DEFAULT_PHONE_CTC_REVISION = "f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99"
NON_SEGMENTAL_TOKENS = frozenset({
    "PAD", "UNK", "SOS", "EOS", "<pad>", "<unk>", "<s>", "</s>",
    "pau", "sil",
})

# For the C-end clarity construct, normal Japanese high-vowel devoicing is a
# legitimate allophonic realization, not a segmental error.  The backend may
# emit uppercase I/U while a target frontend emits lowercase i/u or vice versa.
LOGICAL_ALLOPHONE_MAP = {
    "I": "i",
    "i": "i",
    "U": "u",
    "u": "u",
}


def logical_phone(phone: str) -> str:
    token = str(phone)
    return LOGICAL_ALLOPHONE_MAP.get(token, token)


def sanitize_canonical_phones(phones: Sequence[str]) -> tuple[List[str], List[str]]:
    """Return logical segmental phones plus dropped non-segmental tokens."""
    kept: List[str] = []
    dropped: List[str] = []
    for raw in phones:
        phone = str(raw).strip()
        if not phone:
            continue
        if phone in {"pau", "sil"}:
            dropped.append(phone)
            continue
        kept.append(logical_phone(phone))
    return kept, dropped


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    maxima = np.max(arr, axis=axis, keepdims=True)
    finite = np.isfinite(maxima)
    shifted = np.where(finite, arr - maxima, -np.inf)
    out = maxima + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def project_japanese_ctc_logits(
    logits: np.ndarray,
    vocab: Mapping[str, int],
) -> tuple[np.ndarray, Dict[str, int], Dict[str, List[str]]]:
    """Collapse backend labels into a logical Japanese phone vocabulary.

    The projection is performed in unnormalized-logit space with log-sum-exp,
    which preserves summed softmax mass after a new log-softmax is applied over
    the logical vocabulary.  This prevents i/I and u/U from competing against
    each other in the clarity scorer.
    """
    raw = np.asarray(logits, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("logits must have shape (frames, vocabulary)")
    if not np.isfinite(raw).all():
        raise ValueError("phone-CTC logits contain non-finite values")

    groups: Dict[str, List[Tuple[str, int]]] = {}
    for token, raw_id in vocab.items():
        index = int(raw_id)
        if index < 0 or index >= raw.shape[1]:
            raise ValueError(f"vocabulary id out of model-logit range: {token}={index}")
        logical = logical_phone(str(token))
        groups.setdefault(logical, []).append((str(token), index))

    ordered = sorted(groups.items(), key=lambda item: min(index for _token, index in item[1]))
    logical_vocab: Dict[str, int] = {}
    provenance: Dict[str, List[str]] = {}
    columns: List[np.ndarray] = []
    for new_id, (logical, members) in enumerate(ordered):
        logical_vocab[logical] = new_id
        provenance[logical] = [token for token, _index in members]
        raw_ids = [index for _token, index in members]
        if len(raw_ids) == 1:
            columns.append(raw[:, raw_ids[0]])
        else:
            columns.append(_logsumexp(raw[:, raw_ids], axis=1))

    projected = np.stack(columns, axis=1)
    return projected, logical_vocab, provenance


def segmental_competitor_ids(vocab: Mapping[str, int], *, blank_id: int) -> List[int]:
    """Return only ordinary segmental competitors for clarity evidence."""
    return [
        int(index)
        for token, index in vocab.items()
        if int(index) != int(blank_id) and str(token) not in NON_SEGMENTAL_TOKENS
    ]


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maxima = np.max(values, axis=1, keepdims=True)
    shifted = values - maxima
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _ctc_labels(token_ids: Sequence[int], blank_id: int) -> List[int]:
    labels: List[int] = [int(blank_id)]
    for token_id in token_ids:
        labels.extend([int(token_id), int(blank_id)])
    return labels


def ctc_forward_logprob(
    log_probs: np.ndarray,
    token_ids: Sequence[int],
    *,
    blank_id: int,
) -> float:
    """Exact CTC forward log-probability for one fixed logical phone sequence."""
    probs = np.asarray(log_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        raise ValueError("log_probs must contain at least one CTC frame")
    labels = _ctc_labels(token_ids, int(blank_id))
    states = len(labels)
    frames = probs.shape[0]
    alpha = np.full((frames, states), -np.inf, dtype=np.float64)
    alpha[0, 0] = probs[0, blank_id]
    if states > 1:
        alpha[0, 1] = probs[0, labels[1]]

    def lse(items: Iterable[float]) -> float:
        vals = np.asarray([value for value in items if np.isfinite(value)], dtype=float)
        if vals.size == 0:
            return -np.inf
        m = float(np.max(vals))
        return m + math.log(float(np.sum(np.exp(vals - m))))

    for t in range(1, frames):
        for s in range(states):
            prev = [alpha[t - 1, s]]
            if s - 1 >= 0:
                prev.append(alpha[t - 1, s - 1])
            if s - 2 >= 0 and labels[s] != blank_id and labels[s] != labels[s - 2]:
                prev.append(alpha[t - 1, s - 2])
            prev_sum = lse(prev)
            if np.isfinite(prev_sum):
                alpha[t, s] = prev_sum + probs[t, labels[s]]

    final_states = [states - 1]
    if states > 1:
        final_states.append(states - 2)
    return lse(alpha[-1, state] for state in final_states)


def _collapse_ctc_ids(ids: Sequence[int], *, blank_id: int) -> List[int]:
    out: List[int] = []
    previous: Optional[int] = None
    for value in ids:
        token_id = int(value)
        if token_id == int(blank_id):
            previous = token_id
            continue
        if previous == token_id:
            continue
        out.append(token_id)
        previous = token_id
    return out


def _edit_operations(target: Sequence[str], observed: Sequence[str]) -> tuple[int, List[Dict[str, Any]]]:
    """Levenshtein alignment with explicit insert/delete/substitute operations."""
    a = list(target)
    b = list(observed)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    back: List[List[Optional[str]]] = [[None] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, len(b) + 1):
        dp[0][j] = j
        back[0][j] = "insert"
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                choices = [(dp[i - 1][j - 1], "match")]
            else:
                choices = [
                    (dp[i - 1][j - 1] + 1, "substitute"),
                    (dp[i - 1][j] + 1, "delete"),
                    (dp[i][j - 1] + 1, "insert"),
                ]
            best = min(choices, key=lambda item: item[0])
            dp[i][j], back[i][j] = best

    operations: List[Dict[str, Any]] = []
    i, j = len(a), len(b)
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "match":
            i -= 1
            j -= 1
        elif op == "substitute":
            operations.append({"op": "substitute", "target_index": i - 1, "target": a[i - 1], "observed": b[j - 1]})
            i -= 1
            j -= 1
        elif op == "delete":
            operations.append({"op": "delete", "target_index": i - 1, "target": a[i - 1], "observed": None})
            i -= 1
        elif op == "insert":
            operations.append({"op": "insert", "target_index": i, "target": None, "observed": b[j - 1]})
            j -= 1
        else:
            raise RuntimeError("edit backtrace failed")
    operations.reverse()
    return dp[len(a)][len(b)], operations


def add_sequence_level_evidence(
    result: PhoneGopResult,
    logits: np.ndarray,
    canonical_phones: Sequence[str],
    *,
    vocab: Mapping[str, int],
    blank_id: int,
) -> PhoneGopResult:
    """Attach deletion/insertion-sensitive CTC sequence diagnostics."""
    if not result.available:
        return result
    phones = list(canonical_phones)
    token_ids = [int(vocab[phone]) for phone in phones]
    log_probs = _log_softmax(np.asarray(logits, dtype=np.float64))
    canonical_lp = ctc_forward_logprob(log_probs, token_ids, blank_id=blank_id)
    frame_count = max(int(log_probs.shape[0]), 1)

    deletion_rows: List[Dict[str, Any]] = []
    for index, phone in enumerate(phones):
        deleted = token_ids[:index] + token_ids[index + 1 :]
        deleted_lp = ctc_forward_logprob(log_probs, deleted, blank_id=blank_id)
        deletion_rows.append({
            "phone_index": index,
            "phone": phone,
            "canonical_minus_deleted_logprob_per_frame": float((canonical_lp - deleted_lp) / frame_count),
            "interpretation": "positive_favors_canonical_phone; exploratory_not_calibrated",
        })

    greedy_ids = _collapse_ctc_ids(np.argmax(logits, axis=1).tolist(), blank_id=blank_id)
    id_to_token = {int(index): str(token) for token, index in vocab.items()}
    greedy_phones = [
        id_to_token[token_id]
        for token_id in greedy_ids
        if id_to_token.get(token_id) not in NON_SEGMENTAL_TOKENS
    ]
    edit_distance, operations = _edit_operations(phones, greedy_phones)

    summary = dict(result.summary)
    summary.update({
        "ctc_forward_logprob": float(canonical_lp),
        "ctc_forward_logprob_per_frame": float(canonical_lp / frame_count),
        "leave_one_phone_out_deletion_evidence": deletion_rows,
        "greedy_logical_phone_sequence": greedy_phones,
        "greedy_phone_edit_distance": int(edit_distance),
        "greedy_phone_edit_operations": operations,
        "sequence_evidence_note": (
            "leave_one_out_and_greedy_edits_are_shadow_diagnostics; not a validated deletion/insertion score"
        ),
    })
    return replace(result, summary=summary)


class JapanesePhoneCtcBackend(HuggingFacePhoneCtcBackend):
    """Pinned, Japanese-aware phone-CTC backend for preflight/shadow use."""

    def __init__(
        self,
        model_id: str = DEFAULT_PHONE_CTC_MODEL,
        *,
        revision: str = DEFAULT_PHONE_CTC_REVISION,
        device: Optional[str] = None,
        local_files_only: bool = True,
    ) -> None:
        super().__init__(model_id=model_id, device=device, local_files_only=local_files_only)
        self.revision = revision

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Japanese phone-CTC GOP shadow requires torch and transformers") from exc
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            self.model = AutoModelForCTC.from_pretrained(
                self.model_id,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load pinned phone-CTC model {self.model_id}@{self.revision}: {exc}"
            ) from exc
        self.model.to(self.device)
        self.model.eval()
        self._validate_backend_contract()

    def _expected_sample_rate(self) -> int:
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        value = getattr(feature_extractor, "sampling_rate", None)
        return int(value or 16000)

    def _validate_backend_contract(self) -> None:
        vocab = self.vocabulary()
        config_vocab = int(getattr(self.model.config, "vocab_size", 0) or 0)
        if config_vocab <= 0:
            raise RuntimeError("phone-CTC model does not expose a positive vocab_size")
        if max(vocab.values(), default=-1) >= config_vocab:
            raise RuntimeError("tokenizer vocabulary exceeds model output vocabulary")
        blank_id = int(getattr(self.model.config, "pad_token_id", 0) or 0)
        if blank_id < 0 or blank_id >= config_vocab:
            raise RuntimeError("CTC blank/pad id is outside model vocabulary")
        for required in ("a", "i", "u", "N", "cl", "pau", "sil"):
            if required not in vocab:
                raise RuntimeError(f"expected Japanese phone token missing from backend vocabulary: {required}")
        if self._expected_sample_rate() != 16000:
            raise RuntimeError("default Japanese GOP backend is expected to use 16 kHz audio")

    def _frame_stride_sec(self, audio_length: int, sr: int, frame_count: int) -> float:
        strides = getattr(self.model.config, "conv_stride", None)
        if strides:
            product = 1
            for value in strides:
                product *= int(value)
            if product > 0:
                return float(product) / float(sr)
        return super()._frame_stride_sec(audio_length, sr, frame_count)

    def evaluate(
        self,
        audio: np.ndarray,
        canonical_phones: Sequence[str],
        *,
        sr: int = 16000,
    ) -> PhoneGopResult:
        self._load()
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        clean_phones, dropped = sanitize_canonical_phones(canonical_phones)

        def unavailable(reason: str, warning: str) -> PhoneGopResult:
            return PhoneGopResult(
                available=False,
                backend="japanese_huggingface_phone_ctc",
                model_id=f"{self.model_id}@{self.revision}",
                method="japanese_logical_phone_ctc_evidence_v2",
                canonical_phones=clean_phones,
                evidence=[],
                summary={"reason": reason, "dropped_nonsegmental_target_tokens": dropped},
                warnings=[warning],
            )

        if waveform.size == 0:
            return unavailable("empty_audio", "empty_audio")
        if not np.isfinite(waveform).all():
            return unavailable("nonfinite_audio", "nonfinite_audio")
        if int(sr) != self._expected_sample_rate():
            return unavailable("sampling_rate_mismatch", "sampling_rate_mismatch")
        if waveform.size < int(0.12 * sr):
            return unavailable("audio_too_short", "audio_too_short")
        if float(np.max(np.abs(waveform))) < 1e-5:
            return unavailable("near_silent_audio", "near_silent_audio")
        if not clean_phones:
            return unavailable("empty_segmental_phone_sequence", "empty_phone_sequence")

        inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt")
        model_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            output = self.model(**model_inputs)
        raw_logits = output.logits.squeeze(0).detach().cpu().numpy()
        raw_vocab = self.vocabulary()
        logical_logits, logical_vocab, projection = project_japanese_ctc_logits(raw_logits, raw_vocab)
        if "PAD" not in logical_vocab:
            return unavailable("logical_blank_token_missing", "logical_blank_token_missing")
        blank_id = int(logical_vocab["PAD"])
        missing = [phone for phone in clean_phones if phone not in logical_vocab]
        if missing:
            result = unavailable("canonical_phone_not_in_logical_vocabulary", "phone_inventory_mismatch")
            return replace(result, summary={**result.summary, "missing_phones": sorted(set(missing))})

        frame_stride_sec = self._frame_stride_sec(len(waveform), sr, logical_logits.shape[0])
        competitor_ids = segmental_competitor_ids(logical_vocab, blank_id=blank_id)
        try:
            result = compute_phone_gop_evidence(
                logical_logits,
                clean_phones,
                vocab=logical_vocab,
                blank_id=blank_id,
                frame_stride_sec=frame_stride_sec,
                backend="japanese_huggingface_phone_ctc",
                model_id=f"{self.model_id}@{self.revision}",
                competitor_token_ids=competitor_ids,
            )
        except Exception as exc:
            return unavailable(f"gop_extraction_failed:{type(exc).__name__}", "gop_extraction_failed")

        result = add_sequence_level_evidence(
            result,
            logical_logits,
            clean_phones,
            vocab=logical_vocab,
            blank_id=blank_id,
        )
        summary = dict(result.summary)
        summary.update({
            "model_revision": self.revision,
            "expected_sample_rate": self._expected_sample_rate(),
            "logical_phone_projection": projection,
            "dropped_nonsegmental_target_tokens": dropped,
            "nonsegmental_competitors_excluded": True,
            "high_vowel_allophones_collapsed": True,
            "ctc_support_frames_are_not_physical_phone_boundaries": True,
            "target_frontend_compatibility": "must_be_preflighted_against_pyopenjtalk_plus_training_labels",
        })
        warnings = list(result.warnings)
        if dropped:
            warnings.append("nonsegmental_target_tokens_removed_for_clarity")
        return replace(
            result,
            method="japanese_logical_phone_ctc_evidence_v2",
            summary=summary,
            warnings=warnings,
        )

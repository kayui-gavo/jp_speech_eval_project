"""Japanese-specific hardening for phone-CTC GOP shadow experiments.

This module sits on top of the generic ``phoneme_gop`` feature extractor and
handles Japanese/backend details that must be resolved before human recording:

* collapse legitimate high-vowel devoicing labels (i/I, u/U) into logical
  allophone classes for clarity scoring;
* keep pau/sil and special mora tokens out of **ordinary segmental** competitor
  sets used for clarity evidence;
* pin the default Beatrice v4 model revision for reproducibility;
* validate model/tokenizer/sample-rate compatibility;
* expose sequence-level CTC evidence, including leave-one-phone-out deletion
  preference and greedy edit operations, so a forced canonical Viterbi path is
  not the only signal available for deletion/insertion stress tests.

``N`` and ``cl`` remain valid canonical target tokens and remain visible to
sequence-level diagnostics. They are excluded only from ordinary segmental
competition because Japanese mora nasal / geminate evaluation requires
additional duration/context evidence and is handled primarily by special-mora /
rhythm diagnostics.

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
SPECIAL_MORA_TOKENS = frozenset({"N", "cl"})

# For the C-end clarity construct, normal Japanese high-vowel devoicing is a
# legitimate allophonic realization, not a segmental error. The backend may
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
    """Return logical target phones plus dropped pause/silence tokens.

    ``N`` and ``cl`` are intentionally retained as canonical targets. Their
    presence is linguistically meaningful even though they are excluded from
    the ordinary clarity competitor inventory.
    """
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
    the logical vocabulary. This prevents i/I and u/U from competing against
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
    """Return ordinary segmental competitors for the clarity construct.

    Special mora tokens remain valid target labels but are not ordinary phone
    alternatives. Treating /N/ or /cl/ as a generic substitution competitor
    for vowels/consonants would mix the clarity and timing/special-mora
    constructs that the C-end scoring design intentionally keeps separate.
    """
    return [
        int(index)
        for token, index in vocab.items()
        if int(index) != int(blank_id)
        and str(token) not in NON_SEGMENTAL_TOKENS
        and str(token) not in SPECIAL_MORA_TOKENS
    ]


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    maxima = np.max(logits, axis=1, keepdims=True)
    shifted = logits - maxima
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _logsumexp_list(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return -np.inf
    maximum = max(vals)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in vals))


def ctc_forward_logprob(log_probs: np.ndarray, target_ids: Sequence[int], *, blank_id: int) -> float:
    """Exact CTC sequence log probability for one fixed target sequence."""
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2 or lp.shape[0] <= 0:
        raise ValueError("log_probs must have shape (frames, classes)")
    target = [int(value) for value in target_ids]
    if any(value == int(blank_id) for value in target):
        raise ValueError("target sequence must not contain CTC blank")
    if not target:
        return float(np.sum(lp[:, int(blank_id)]))
    extended: List[int] = [int(blank_id)]
    for token in target:
        extended.extend([token, int(blank_id)])
    states = len(extended)
    previous = np.full(states, -np.inf, dtype=np.float64)
    previous[0] = lp[0, int(blank_id)]
    if states > 1:
        previous[1] = lp[0, extended[1]]
    for frame in range(1, lp.shape[0]):
        current = np.full(states, -np.inf, dtype=np.float64)
        for state, token in enumerate(extended):
            predecessors = [previous[state]]
            if state > 0:
                predecessors.append(previous[state - 1])
            if (
                state > 1
                and token != int(blank_id)
                and token != extended[state - 2]
            ):
                predecessors.append(previous[state - 2])
            current[state] = _logsumexp_list(predecessors) + lp[frame, token]
        previous = current
    return _logsumexp_list(previous[-2:])


def ctc_viterbi_phone_spans(
    log_probs: np.ndarray,
    target_ids: Sequence[int],
    *,
    blank_id: int,
) -> Optional[List[Tuple[int, int]]]:
    """Best-path CTC support spans for target phones.

    These spans are alignment support regions, not gold acoustic boundaries.
    They are useful for research diagnostics but must not be interpreted as
    physical phone durations without a separate alignment validation.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    target = [int(value) for value in target_ids]
    if not target:
        return []
    if any(value == int(blank_id) for value in target):
        return None
    extended: List[int] = [int(blank_id)]
    phone_state_to_index: Dict[int, int] = {}
    for phone_index, token in enumerate(target):
        phone_state = len(extended)
        extended.extend([token, int(blank_id)])
        phone_state_to_index[phone_state] = phone_index
    states = len(extended)
    frames = lp.shape[0]
    dp = np.full((frames, states), -np.inf, dtype=np.float64)
    back = np.full((frames, states), -1, dtype=np.int32)
    dp[0, 0] = lp[0, blank_id]
    if states > 1:
        dp[0, 1] = lp[0, extended[1]]
    for frame in range(1, frames):
        for state, token in enumerate(extended):
            candidates = [(dp[frame - 1, state], state)]
            if state > 0:
                candidates.append((dp[frame - 1, state - 1], state - 1))
            if (
                state > 1
                and token != blank_id
                and token != extended[state - 2]
            ):
                candidates.append((dp[frame - 1, state - 2], state - 2))
            best_score, best_state = max(candidates, key=lambda item: item[0])
            dp[frame, state] = best_score + lp[frame, token]
            back[frame, state] = best_state
    final_candidates = [(dp[-1, states - 1], states - 1)]
    if states > 1:
        final_candidates.append((dp[-1, states - 2], states - 2))
    final_score, state = max(final_candidates, key=lambda item: item[0])
    if not np.isfinite(final_score):
        return None
    path_states = [state]
    for frame in range(frames - 1, 0, -1):
        state = int(back[frame, state])
        if state < 0:
            return None
        path_states.append(state)
    path_states.reverse()
    spans: List[Tuple[int, int]] = []
    for phone_index in range(len(target)):
        matching_states = [
            state for state, idx in phone_state_to_index.items() if idx == phone_index
        ]
        positions = [
            frame for frame, state in enumerate(path_states) if state in matching_states
        ]
        if not positions:
            return None
        spans.append((min(positions), max(positions) + 1))
    return spans


def _greedy_sequence(logits: np.ndarray, vocab: Mapping[str, int], *, blank_id: int) -> List[str]:
    inverse = {int(index): logical_phone(str(token)) for token, index in vocab.items()}
    ids = np.argmax(np.asarray(logits), axis=1).tolist()
    result: List[str] = []
    previous: Optional[int] = None
    for raw_id in ids:
        idx = int(raw_id)
        if previous == idx:
            continue
        previous = idx
        if idx == int(blank_id):
            continue
        token = inverse.get(idx)
        if token and token not in NON_SEGMENTAL_TOKENS:
            result.append(token)
    return result


def _levenshtein_ops(target: Sequence[str], observed: Sequence[str]) -> List[Dict[str, Any]]:
    """Phone-sequence edits from target to unconstrained greedy observation."""
    a = list(target)
    b = list(observed)
    rows = len(a) + 1
    cols = len(b) + 1
    dp = np.zeros((rows, cols), dtype=np.int32)
    back: List[List[Optional[str]]] = [[None for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        dp[i, 0] = i
        back[i][0] = "delete"
    for j in range(1, cols):
        dp[0, j] = j
        back[0][j] = "insert"
    for i in range(1, rows):
        for j in range(1, cols):
            if a[i - 1] == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
                back[i][j] = "match"
                continue
            candidates = [
                (dp[i - 1, j - 1] + 1, "substitute"),
                (dp[i - 1, j] + 1, "delete"),
                (dp[i, j - 1] + 1, "insert"),
            ]
            cost, op = min(candidates, key=lambda item: item[0])
            dp[i, j] = cost
            back[i][j] = op
    ops: List[Dict[str, Any]] = []
    i, j = len(a), len(b)
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "match":
            i -= 1
            j -= 1
        elif op == "substitute":
            ops.append({"type": "substitution", "target_index": i - 1, "target": a[i - 1], "observed": b[j - 1]})
            i -= 1
            j -= 1
        elif op == "delete":
            ops.append({"type": "deletion", "target_index": i - 1, "target": a[i - 1], "observed": None})
            i -= 1
        elif op == "insert":
            ops.append({"type": "insertion", "target_index": i, "target": None, "observed": b[j - 1]})
            j -= 1
        else:
            break
    ops.reverse()
    return ops


def sequence_level_evidence(
    logits: np.ndarray,
    canonical_phones: Sequence[str],
    *,
    vocab: Mapping[str, int],
    blank_id: int,
) -> Dict[str, Any]:
    """Alignment-free sequence/deletion evidence; no learner score mapping."""
    phones, dropped = sanitize_canonical_phones(canonical_phones)
    missing = [phone for phone in phones if phone not in vocab]
    if missing:
        return {"available": False, "reason": "canonical_phone_not_in_vocabulary", "missing_phones": missing}
    log_probs = _log_softmax(np.asarray(logits, dtype=np.float64))
    ids = [int(vocab[phone]) for phone in phones]
    canonical_lp = ctc_forward_logprob(log_probs, ids, blank_id=blank_id)
    deletion_rows: List[Dict[str, Any]] = []
    for index, phone in enumerate(phones):
        deleted = ids[:index] + ids[index + 1 :]
        deleted_lp = ctc_forward_logprob(log_probs, deleted, blank_id=blank_id)
        deletion_rows.append(
            {
                "phone_index": index,
                "canonical_phone": phone,
                "canonical_logprob": canonical_lp,
                "deleted_target_logprob": deleted_lp,
                "canonical_minus_deleted": canonical_lp - deleted_lp,
                "deletion_preferred_over_canonical": bool(deleted_lp > canonical_lp),
            }
        )
    greedy = _greedy_sequence(logits, vocab, blank_id=blank_id)
    return {
        "available": True,
        "canonical_logprob": canonical_lp,
        "deletion_evidence": deletion_rows,
        "greedy_phone_sequence": greedy,
        "greedy_edit_operations": _levenshtein_ops(phones, greedy),
        "dropped_nonsegmental_target_tokens": dropped,
        "score_mapped": False,
        "interpretation": "alignment_free_sequence_support_not_pronunciation_score",
    }


class JapanesePhoneCtcBackend(HuggingFacePhoneCtcBackend):
    """Pinned Japanese phone CTC backend with logical allophone projection."""

    def __init__(
        self,
        model_id: str = DEFAULT_PHONE_CTC_MODEL,
        *,
        revision: str = DEFAULT_PHONE_CTC_REVISION,
        device: Optional[str] = None,
        local_files_only: bool = True,
    ):
        super().__init__(model_id=model_id, device=device, local_files_only=local_files_only)
        self.revision = str(revision)

    def _load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for phone GOP") from exc
        kwargs: Dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "revision": self.revision,
        }
        self.processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        self.model = AutoModelForCTC.from_pretrained(self.model_id, **kwargs)
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, audio: np.ndarray, phones: Sequence[str], *, sr: int = 16000) -> PhoneGopResult:
        self._load()
        if int(sr) != 16000:
            return PhoneGopResult(
                available=False,
                backend="huggingface_phone_ctc",
                model_id=f"{self.model_id}@{self.revision}",
                method="ctc_viterbi_phone_evidence_v1",
                canonical_phones=list(phones),
                evidence=[],
                summary={"reason": "sample_rate_must_be_16000"},
                warnings=["sample_rate_mismatch"],
            )
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size < 160:
            return PhoneGopResult(
                available=False,
                backend="huggingface_phone_ctc",
                model_id=f"{self.model_id}@{self.revision}",
                method="ctc_viterbi_phone_evidence_v1",
                canonical_phones=list(phones),
                evidence=[],
                summary={"reason": "audio_too_short"},
                warnings=["audio_too_short"],
            )
        inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt")
        model_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            output = self.model(**model_inputs)
        raw_logits = output.logits.squeeze(0).detach().cpu().numpy()
        raw_vocab = self.vocabulary()
        logical_logits, logical_vocab, provenance = project_japanese_ctc_logits(raw_logits, raw_vocab)
        if "PAD" not in logical_vocab:
            return PhoneGopResult(
                available=False,
                backend="huggingface_phone_ctc",
                model_id=f"{self.model_id}@{self.revision}",
                method="ctc_viterbi_phone_evidence_v1",
                canonical_phones=list(phones),
                evidence=[],
                summary={"reason": "logical_blank_token_missing"},
                warnings=["logical_blank_token_missing"],
            )
        clean_phones, dropped = sanitize_canonical_phones(phones)
        result = compute_phone_gop_evidence(
            logical_logits,
            clean_phones,
            vocab=logical_vocab,
            blank_id=int(logical_vocab["PAD"]),
            frame_stride_sec=None,
            audio_duration_sec=float(waveform.size) / sr,
            excluded_competitor_tokens=NON_SEGMENTAL_TOKENS | SPECIAL_MORA_TOKENS,
            backend="huggingface_phone_ctc",
            model_id=f"{self.model_id}@{self.revision}",
        )
        seq = sequence_level_evidence(
            logical_logits,
            clean_phones,
            vocab=logical_vocab,
            blank_id=int(logical_vocab["PAD"]),
        )
        summary = dict(result.summary)
        summary.update(
            {
                "revision": self.revision,
                "logical_allophone_groups": {
                    phone: members for phone, members in provenance.items() if len(members) > 1
                },
                "dropped_nonsegmental_target_tokens": dropped,
                "sequence_evidence": seq,
                "special_mora_excluded_from_ordinary_competitors": True,
                "ordinary_competitor_tokens_exclude": sorted(NON_SEGMENTAL_TOKENS | SPECIAL_MORA_TOKENS),
            }
        )
        warnings = list(result.warnings)
        return replace(result, summary=summary, warnings=warnings)

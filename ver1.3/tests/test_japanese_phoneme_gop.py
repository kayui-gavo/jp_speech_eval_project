from __future__ import annotations

import numpy as np

from jp_speech_eval.japanese_phoneme_gop import (
    NON_SEGMENTAL_TOKENS,
    add_sequence_level_evidence,
    ctc_forward_logprob,
    project_japanese_ctc_logits,
    sanitize_canonical_phones,
    segmental_competitor_ids,
)
from jp_speech_eval.phoneme_gop import compute_phone_gop_evidence


def test_high_vowel_devoicing_labels_are_collapsed_not_competitors() -> None:
    # raw vocab: PAD, i, I, u, U, k, pau, sil
    vocab = {"PAD": 0, "i": 1, "I": 2, "u": 3, "U": 4, "k": 5, "pau": 6, "sil": 7}
    logits = np.full((5, 8), -5.0, dtype=float)
    logits[:, 0] = 0.0
    logits[1, 2] = 8.0  # backend strongly prefers devoiced I
    logits[2, 0] = 7.0
    logits[3, 5] = 8.0

    projected, logical_vocab, provenance = project_japanese_ctc_logits(logits, vocab)
    assert "I" not in logical_vocab
    assert "U" not in logical_vocab
    assert set(provenance["i"]) == {"i", "I"}
    assert set(provenance["u"]) == {"u", "U"}

    result = compute_phone_gop_evidence(
        projected,
        ["i", "k"],
        vocab=logical_vocab,
        blank_id=logical_vocab["PAD"],
        frame_stride_sec=0.02,
        competitor_token_ids=segmental_competitor_ids(
            logical_vocab, blank_id=logical_vocab["PAD"]
        ),
    )
    assert result.available is True
    by_phone = {row.canonical_phone: row for row in result.evidence}
    assert by_phone["i"].posterior_gop_margin > 0


def test_pause_and_silence_are_not_segmental_competitors() -> None:
    vocab = {"PAD": 0, "a": 1, "k": 2, "pau": 3, "sil": 4, "UNK": 5}
    competitor_ids = segmental_competitor_ids(vocab, blank_id=0)
    assert vocab["a"] in competitor_ids
    assert vocab["k"] in competitor_ids
    for token in NON_SEGMENTAL_TOKENS:
        if token in vocab:
            assert vocab[token] not in competitor_ids


def test_target_pauses_are_removed_from_clarity_sequence() -> None:
    kept, dropped = sanitize_canonical_phones(["k", "o", "pau", "N", "sil"])
    assert kept == ["k", "o", "N"]
    assert dropped == ["pau", "sil"]


def _deletion_logits() -> tuple[np.ndarray, dict[str, int]]:
    # Logical vocab PAD, a, b, c. Acoustics strongly support a-c with no b.
    vocab = {"PAD": 0, "a": 1, "b": 2, "c": 3}
    logits = np.full((8, 4), -5.0, dtype=float)
    logits[:, 0] = 0.0
    logits[0, 0] = 8.0
    logits[1:3, 1] = 9.0
    logits[3:5, 0] = 8.0
    logits[5:7, 3] = 9.0
    logits[7, 0] = 8.0
    return logits, vocab


def test_leave_one_phone_out_evidence_detects_missing_middle_phone() -> None:
    logits, vocab = _deletion_logits()
    base = compute_phone_gop_evidence(
        logits,
        ["a", "b", "c"],
        vocab=vocab,
        blank_id=0,
        frame_stride_sec=0.02,
        competitor_token_ids=[1, 2, 3],
    )
    enriched = add_sequence_level_evidence(
        base,
        logits,
        ["a", "b", "c"],
        vocab=vocab,
        blank_id=0,
    )
    rows = enriched.summary["leave_one_phone_out_deletion_evidence"]
    by_phone = {row["phone"]: row for row in rows}
    # Negative means the audio is better explained when that canonical phone is
    # removed. The deliberately absent /b/ should be the strongest such case.
    b_margin = by_phone["b"]["canonical_minus_deleted_logprob_per_frame"]
    a_margin = by_phone["a"]["canonical_minus_deleted_logprob_per_frame"]
    c_margin = by_phone["c"]["canonical_minus_deleted_logprob_per_frame"]
    assert b_margin < a_margin
    assert b_margin < c_margin
    assert enriched.summary["greedy_phone_edit_distance"] >= 1
    assert any(
        op["op"] in {"delete", "substitute"}
        for op in enriched.summary["greedy_phone_edit_operations"]
    )


def test_sequence_evidence_does_not_create_product_score_mapping() -> None:
    logits, vocab = _deletion_logits()
    base = compute_phone_gop_evidence(
        logits,
        ["a", "b", "c"],
        vocab=vocab,
        blank_id=0,
        frame_stride_sec=0.02,
        competitor_token_ids=[1, 2, 3],
    )
    enriched = add_sequence_level_evidence(base, logits, ["a", "b", "c"], vocab=vocab, blank_id=0)
    assert enriched.score_mapped is False
    assert enriched.product_calibrated is False


def test_ctc_forward_probability_handles_repeated_phones() -> None:
    # CTC repeated labels require an intervening blank; the forward algorithm
    # must keep that topology rather than silently collapsing /a a/ into /a/.
    logits = np.full((7, 3), -6.0, dtype=float)
    logits[:, 0] = 0.0  # blank
    logits[1:3, 1] = 8.0
    logits[3, 0] = 8.0
    logits[4:6, 1] = 8.0
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    repeated = ctc_forward_logprob(log_probs, [1, 1], blank_id=0)
    single = ctc_forward_logprob(log_probs, [1], blank_id=0)
    assert np.isfinite(repeated)
    assert np.isfinite(single)
    assert repeated > single


def test_greedy_sequence_evidence_localizes_an_inserted_phone() -> None:
    vocab = {"PAD": 0, "a": 1, "b": 2, "c": 3}
    # Observed acoustics are a -> c -> b, while canonical target is a -> b.
    logits = np.full((10, 4), -6.0, dtype=float)
    logits[:, 0] = 0.0
    logits[0, 0] = 8.0
    logits[1:3, 1] = 9.0
    logits[3, 0] = 8.0
    logits[4:6, 3] = 9.0
    logits[6, 0] = 8.0
    logits[7:9, 2] = 9.0
    logits[9, 0] = 8.0
    base = compute_phone_gop_evidence(
        logits,
        ["a", "b"],
        vocab=vocab,
        blank_id=0,
        frame_stride_sec=0.02,
        competitor_token_ids=[1, 2, 3],
    )
    enriched = add_sequence_level_evidence(base, logits, ["a", "b"], vocab=vocab, blank_id=0)
    assert enriched.summary["greedy_logical_phone_sequence"] == ["a", "c", "b"]
    operations = enriched.summary["greedy_phone_edit_operations"]
    assert any(op["op"] == "insert" and op["observed"] == "c" for op in operations)

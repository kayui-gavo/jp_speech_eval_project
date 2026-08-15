"""Japanese phone-inventory policies for research diagnostics.

The project intentionally uses two different token sets:

* ``ordinary clarity competitors`` exclude special morae ``N``/``cl``;
* ``acoustic phone tokens`` include ``N``/``cl`` but exclude CTC blank,
  tokenizer controls and pause/silence labels.

Reusing the ordinary clarity competitor set for CTC posterior peakiness would
under-count phone probability mass whenever the model emits a special mora.
This module makes the broader acoustic inventory explicit without changing the
clarity competitor policy in :mod:`japanese_phoneme_gop`.
"""

from __future__ import annotations

from typing import List, Mapping

from .japanese_phoneme_gop import NON_SEGMENTAL_TOKENS, SPECIAL_MORA_TOKENS


def acoustic_phone_token_ids(vocab: Mapping[str, int], *, blank_id: int) -> List[int]:
    """Return all acoustic phone-event token ids, including ``N`` and ``cl``."""
    ids = [
        int(index)
        for token, index in vocab.items()
        if int(index) != int(blank_id)
        and str(token) not in NON_SEGMENTAL_TOKENS
    ]
    if not ids:
        raise ValueError("acoustic phone token inventory is empty")
    return sorted(set(ids))


def acoustic_phone_tokens(vocab: Mapping[str, int], *, blank_id: int) -> List[str]:
    """Return acoustic phone token names in token-id order."""
    ids = set(acoustic_phone_token_ids(vocab, blank_id=blank_id))
    ordered = sorted(
        ((int(index), str(token)) for token, index in vocab.items() if int(index) in ids),
        key=lambda item: (item[0], item[1]),
    )
    return [token for _index, token in ordered]


def inventory_semantics(vocab: Mapping[str, int], *, blank_id: int) -> dict:
    tokens = acoustic_phone_tokens(vocab, blank_id=blank_id)
    return {
        "inventory_role": "all_acoustic_phone_events_for_model_posterior_diagnostics",
        "ordinary_clarity_competitor_inventory": False,
        "includes_special_mora_N": "N" in tokens,
        "includes_special_mora_cl": "cl" in tokens,
        "excludes_pause_and_control_tokens": True,
        "tokens": tokens,
        "token_count": len(tokens),
        "special_mora_tokens": sorted(SPECIAL_MORA_TOKENS),
    }

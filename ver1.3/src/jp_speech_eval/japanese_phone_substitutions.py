"""Japanese phone-substitution search spaces for research GOP/MDD.

This module defines *candidate search spaces*, not learner error labels.
The restricted inventory is deliberately auditable and conservative: it keeps
phones that are phonologically close in Japanese and a small number of
well-motivated diagnostic contrasts used by the project's pilot protocol.

The motivation follows recent alignment-free GOP work showing that restricting
substitution alternatives with phonological knowledge can improve efficiency
and MDD performance versus unrestricted substitution search.  That evidence is
from L2 English; therefore this Japanese inventory remains a research
hypothesis until it is validated on labeled Japanese learner speech.

Special morae ``N`` and ``cl`` are not ordinary segmental substitution targets.
They require duration/context evidence and are handled outside this inventory.
Likewise ``pau``/``sil`` and the CTC blank are never segmental alternatives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence


POLICY_NAME = "restricted_japanese_phonology_v1"
POLICY_STATUS = "research_hypothesis_not_learner_validated"
NONSEGMENTAL_TOKENS = frozenset({"PAD", "UNK", "SOS", "EOS", "pau", "sil", "<pad>", "<unk>", "<s>", "</s>"})
SPECIAL_MORA_TOKENS = frozenset({"N", "cl"})
ALLOPHONE_EQUIVALENCE = {
    "i": frozenset({"i", "I"}),
    "I": frozenset({"i", "I"}),
    "u": frozenset({"u", "U"}),
    "U": frozenset({"u", "U"}),
}

# Phone-level neighborhoods, not kana-level spelling confusions.  Entries are
# intentionally symmetric after normalization below.
_SEED_NEIGHBORS: Dict[str, set[str]] = {
    # Vowels: all Japanese vowel categories are retained as substitution
    # competitors.  The inventory is small and vowel-category errors need not
    # be forced into an arbitrary one-dimensional proximity order.
    "a": {"i", "u", "e", "o"},
    "i": {"a", "u", "e", "o"},
    "u": {"a", "i", "e", "o"},
    "e": {"a", "i", "u", "o"},
    "o": {"a", "i", "u", "e"},

    # Voicing / laryngeal contrasts.
    "k": {"g", "ky"},
    "g": {"k", "gy"},
    "ky": {"gy", "k"},
    "gy": {"ky", "g"},
    "s": {"z", "sh", "ts"},
    "z": {"s", "j"},
    "t": {"d", "ts", "ch", "ty"},
    "d": {"t", "dy"},
    "b": {"p", "by"},
    "p": {"b", "py"},
    "by": {"py", "b"},
    "py": {"by", "p"},

    # Fricative/affricate and palatalization neighborhoods.
    "sh": {"s", "ch", "j"},
    "ts": {"s", "t", "ch"},
    "ch": {"sh", "ts", "j", "t"},
    "j": {"sh", "ch", "z"},
    "h": {"f", "hy"},
    "f": {"h"},
    "hy": {"h"},
    "n": {"ny"},
    "ny": {"n"},
    "m": {"my"},
    "my": {"m"},
    "r": {"ry", "d"},
    "ry": {"r"},
    "w": {"y"},
    "y": {"w"},
    "ty": {"t", "dy", "ch"},
    "dy": {"d", "ty", "j"},
}


def _symmetrize(seed: Mapping[str, Iterable[str]]) -> Dict[str, frozenset[str]]:
    work: Dict[str, set[str]] = {str(phone): {str(x) for x in values} for phone, values in seed.items()}
    for phone, values in list(work.items()):
        for value in values:
            work.setdefault(value, set()).add(phone)
    return {phone: frozenset(sorted(values - {phone})) for phone, values in work.items()}


RESTRICTED_NEIGHBORS = _symmetrize(_SEED_NEIGHBORS)


def canonical_logical_phone(phone: str) -> str:
    """Collapse model-only high-vowel voicing labels to one logical phone."""
    value = str(phone)
    if value == "I":
        return "i"
    if value == "U":
        return "u"
    return value


def is_segmental_phone(phone: str) -> bool:
    value = str(phone)
    return value not in NONSEGMENTAL_TOKENS and value not in SPECIAL_MORA_TOKENS


@dataclass(frozen=True)
class SubstitutionCandidateSet:
    canonical_phone: str
    logical_phone: str
    candidates: tuple[str, ...]
    policy: str
    status: str
    fallback_used: bool
    fallback_reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def restricted_substitution_phones(
    canonical_phone: str,
    available_phones: Sequence[str],
    *,
    include_canonical: bool = True,
    fallback_to_unrestricted: bool = True,
) -> SubstitutionCandidateSet:
    """Return Japanese phonology-informed substitution alternatives.

    The returned set only contains phones present in ``available_phones``.
    ``N``/``cl`` and nonsegmental tokens are always excluded.  If a target has
    no curated neighborhood, callers may explicitly fall back to the full
    segmental inventory; this is recorded in provenance rather than hidden.
    """
    original = str(canonical_phone)
    logical = canonical_logical_phone(original)
    available = {
        canonical_logical_phone(phone)
        for phone in available_phones
        if is_segmental_phone(canonical_logical_phone(phone))
    }
    curated = set(RESTRICTED_NEIGHBORS.get(logical, ())) & available
    fallback_used = False
    fallback_reason = None
    policy = POLICY_NAME
    if not curated and fallback_to_unrestricted:
        curated = set(available) - {logical}
        fallback_used = True
        fallback_reason = "no_curated_neighbors_in_backend_inventory"
        policy = "unrestricted_segmental_fallback"
    if include_canonical and logical in available:
        curated.add(logical)
    curated.discard("N")
    curated.discard("cl")
    return SubstitutionCandidateSet(
        canonical_phone=original,
        logical_phone=logical,
        candidates=tuple(sorted(curated)),
        policy=policy,
        status=POLICY_STATUS,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


def substitution_token_ids_by_position(
    canonical_phones: Sequence[str],
    vocab: Mapping[str, int],
    *,
    fallback_to_unrestricted: bool = True,
) -> tuple[Dict[int, list[int]], list[Dict[str, Any]]]:
    """Build per-position token-id restrictions plus auditable provenance."""
    available = list(vocab.keys())
    token_ids: Dict[int, list[int]] = {}
    provenance: list[Dict[str, Any]] = []
    for index, phone in enumerate(canonical_phones):
        result = restricted_substitution_phones(
            phone,
            available,
            include_canonical=True,
            fallback_to_unrestricted=fallback_to_unrestricted,
        )
        ids = [int(vocab[p]) for p in result.candidates if p in vocab]
        token_ids[int(index)] = sorted(set(ids))
        row = result.to_dict()
        row["phone_index"] = int(index)
        row["candidate_count"] = len(token_ids[int(index)])
        provenance.append(row)
    return token_ids, provenance

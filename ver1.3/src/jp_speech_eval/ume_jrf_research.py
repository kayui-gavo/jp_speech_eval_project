"""Research-only UME-JRF criterion schema and fail-closed layout probing.

UME-JRF is an unusually useful Japanese learner-speech criterion corpus, but
its official license is research-only / non-commercial.  This module therefore
makes licensing and construct provenance first-class data instead of allowing
corpus labels to leak into the commercial C-end runtime accidentally.

The public corpus introduction describes the *conceptual* grading design and
points to corpus-internal files such as ``Vol1/doc/FJlabel/description.txt``.
It does not publicly specify every on-disk grading-table field.  Accordingly,
this module does not guess a label parser.  ``probe_ume_jrf_layout`` inventories
what is actually present; parsing can be implemented only after the real
``FJlabel`` documentation has been inspected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


CORPUS = "UME-JRF"
DATA_DOI = "10.32130/src.UME-JRF"
OFFICIAL_PAGE = "https://research.nii.ac.jp/src/en/UME-JRF.html"
OFFICIAL_INTRODUCTION = "https://research.nii.ac.jp/src/files/UME-JRF.pdf"
LICENSE = "research_only_noncommercial"

SET_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "A": {
        "construct": "broad_pronunciation_quality_relative_to_ideal_japanese_speaker",
        "scale": "ordinal_1_5",
        "rated_subset_public_description": "first_5_sentences_of_assigned_50_or_53_sentence_list",
    },
    "B": {
        "construct": "item_specific_difficult_sound_or_minimal_pair_correctness",
        "scale": "binary_correctness",
        "rated_subset_public_description": "28_or_29_of_assigned_54_difficult_sound_sentences",
    },
    "C": {
        "construct": "item_specific_prosody_correctness_or_naturalness",
        "scale": "ordinal_1_5",
        "rated_subset_public_description": "12_of_42_prosody_sentences",
    },
    "D": {
        "construct": "item_specific_target_phone_correctness",
        "scale": "ordinal_1_5",
        "rated_subset_public_description": "10_of_115_difficult_sound_words",
    },
}

D_RATED_WORDS = (
    "酸っぱい",
    "全員",
    "王座",
    "通信",
    "カミュ",
    "廊下",
    "友情",
    "ビル",
    "美容院",
    "合唱",
)

KNOWN_DOCUMENT_PATH_SUFFIXES = (
    Path("Vol1/doc/FJcontent/description.txt"),
    Path("Vol1/doc/FJlabel/description.txt"),
)


@dataclass(frozen=True)
class UmeJrfCriterionLabel:
    corpus: str
    set_id: str
    construct: str
    scale: str
    item_id: str
    rater_id: str
    raw_label: int
    target_description: Optional[str]
    source_label_file: Optional[str]
    normalized_100: None = None
    research_only_license: bool = True
    commercial_product_use_allowed: bool = False
    product_score_mapped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UmeJrfLayoutProbe:
    available: bool
    corpus_root: str
    license: str
    research_only_license: bool
    commercial_product_use_allowed: bool
    known_documents: Dict[str, Dict[str, Any]]
    candidate_label_files: list[str]
    candidate_audio_files: list[str]
    grading_schema_status: str
    automatic_label_parsing_allowed: bool
    notes: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_raw_label(scale: str, value: int) -> bool:
    if scale == "ordinal_1_5":
        return int(value) in {1, 2, 3, 4, 5}
    if scale == "binary_correctness":
        return int(value) in {0, 1}
    return False


def build_ume_jrf_criterion_label(
    *,
    set_id: str,
    item_id: str,
    rater_id: str,
    raw_label: int,
    target_description: Optional[str] = None,
    source_label_file: Optional[str] = None,
) -> UmeJrfCriterionLabel:
    """Create one typed raw expert label after corpus-specific parsing.

    This function validates a label already extracted according to the actual
    corpus documentation.  It is intentionally *not* a generic numeric-column
    parser and never converts an ordinal/binary raw label to `/100`.
    """
    set_key = str(set_id).strip().upper()
    if set_key not in SET_DEFINITIONS:
        raise ValueError(f"unsupported UME-JRF set: {set_id!r}")
    item = str(item_id).strip()
    rater = str(rater_id).strip()
    if not item:
        raise ValueError("UME-JRF criterion label requires item_id")
    if not rater:
        raise ValueError("UME-JRF criterion label requires pseudonymous rater_id")
    definition = SET_DEFINITIONS[set_key]
    scale = str(definition["scale"])
    value = int(raw_label)
    if not _valid_raw_label(scale, value):
        raise ValueError(f"raw label {value!r} is invalid for UME-JRF set {set_key} scale {scale}")
    return UmeJrfCriterionLabel(
        corpus=CORPUS,
        set_id=set_key,
        construct=str(definition["construct"]),
        scale=scale,
        item_id=item,
        rater_id=rater,
        raw_label=value,
        target_description=(str(target_description) if target_description is not None else None),
        source_label_file=(str(source_label_file) if source_label_file is not None else None),
        normalized_100=None,
        research_only_license=True,
        commercial_product_use_allowed=False,
        product_score_mapped=False,
    )


def _limited_candidates(root: Path, patterns: Iterable[str], *, limit: int = 500) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            try:
                relative = str(path.relative_to(root))
            except ValueError:
                relative = str(path)
            found.append(relative)
            if len(found) >= limit:
                return sorted(set(found))
    return sorted(set(found))


def probe_ume_jrf_layout(corpus_root: str | Path) -> UmeJrfLayoutProbe:
    """Inspect a local UME-JRF checkout without guessing its grading schema."""
    root = Path(corpus_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return UmeJrfLayoutProbe(
            available=False,
            corpus_root=str(root),
            license=LICENSE,
            research_only_license=True,
            commercial_product_use_allowed=False,
            known_documents={},
            candidate_label_files=[],
            candidate_audio_files=[],
            grading_schema_status="corpus_root_missing",
            automatic_label_parsing_allowed=False,
            notes=["No local UME-JRF directory was found; nothing was downloaded."],
        )

    documents: Dict[str, Dict[str, Any]] = {}
    for suffix in KNOWN_DOCUMENT_PATH_SUFFIXES:
        path = root / suffix
        if path.is_file():
            documents[str(suffix)] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "public_introduction_points_here": True,
            }

    # These are candidates only. Their formats/semantics are not inferred.
    label_candidates = _limited_candidates(
        root,
        ("*label*", "*Label*", "*.lab", "*.csv", "*.tsv"),
    )
    audio_candidates = _limited_candidates(root, ("*.wav", "*.WAV"))
    label_description_present = "Vol1/doc/FJlabel/description.txt" in documents

    notes = [
        "Candidate filenames are inventory only; no unknown numeric field is treated as a pronunciation label.",
        "Inspect Vol1/doc/FJlabel/description.txt before implementing a concrete parser.",
        "UME-JRF is research-only/non-commercial and must not be copied into the product runtime.",
    ]
    status = (
        "label_documentation_present_requires_human_schema_inspection"
        if label_description_present
        else "label_documentation_not_found_parser_blocked"
    )
    return UmeJrfLayoutProbe(
        available=True,
        corpus_root=str(root),
        license=LICENSE,
        research_only_license=True,
        commercial_product_use_allowed=False,
        known_documents=documents,
        candidate_label_files=label_candidates,
        candidate_audio_files=audio_candidates,
        grading_schema_status=status,
        automatic_label_parsing_allowed=False,
        notes=notes,
    )


def research_guard_metadata() -> Dict[str, Any]:
    """Machine-readable license/construct guard for downstream reports."""
    return {
        "corpus": CORPUS,
        "data_doi": DATA_DOI,
        "official_page": OFFICIAL_PAGE,
        "official_introduction": OFFICIAL_INTRODUCTION,
        "license": LICENSE,
        "research_only_license": True,
        "commercial_product_use_allowed": False,
        "product_runtime_ingestion_allowed": False,
        "product_model_training_allowed_without_separate_permission": False,
        "normalize_raw_expert_labels_to_100_on_import": False,
        "set_definitions": SET_DEFINITIONS,
        "d_rated_words": list(D_RATED_WORDS),
    }

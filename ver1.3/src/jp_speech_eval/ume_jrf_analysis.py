"""Analysis scaffolding for typed UME-JRF expert criterion labels.

This module intentionally starts *after* corpus-specific label parsing. It never
reads unknown UME-JRF files, guesses numeric columns, normalizes ratings to
``/100`` or trains a product model. Its job is to preserve raw rater evidence,
aggregate criterion units transparently, report coverage, and build speaker-
and item-held-out validation folds before any phone feature is fitted.

The functions are generic across the four UME-JRF sets but never pool different
constructs/scales into one criterion. Set C prosody therefore cannot be mixed
with B/D segmental correctness accidentally.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import statistics
from typing import Any, Dict, Iterable, List, Literal, Sequence

from .ume_jrf_research import UmeJrfCriterionLabel


@dataclass(frozen=True)
class UmeJrfCriterionUnit:
    set_id: str
    construct: str
    scale: str
    speaker_id: str
    item_id: str
    rater_ids: List[str]
    raw_labels: List[int]
    rater_count: int
    ordinal_median: float | None
    binary_correct_fraction: float | None
    binary_majority_label: int | None
    target_description: str | None
    normalized_100: None = None
    product_score_mapped: bool = False
    research_only_license: bool = True
    commercial_product_use_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_single_construct(labels: Sequence[UmeJrfCriterionLabel]) -> tuple[str, str, str]:
    if not labels:
        raise ValueError("criterion analysis requires at least one expert label")
    sets = {label.set_id for label in labels}
    constructs = {label.construct for label in labels}
    scales = {label.scale for label in labels}
    if len(sets) != 1 or len(constructs) != 1 or len(scales) != 1:
        raise ValueError(
            "UME-JRF criterion analysis must not pool different sets/constructs/scales"
        )
    return next(iter(sets)), next(iter(constructs)), next(iter(scales))


def aggregate_criterion_units(
    labels: Iterable[UmeJrfCriterionLabel],
) -> List[UmeJrfCriterionUnit]:
    """Aggregate raw rater labels per speaker×item without losing raw evidence."""
    label_list = list(labels)
    set_id, construct, scale = _validate_single_construct(label_list)
    grouped: dict[tuple[str, str], list[UmeJrfCriterionLabel]] = defaultdict(list)
    for label in label_list:
        if label.set_id != set_id or label.construct != construct or label.scale != scale:
            raise ValueError("mixed criterion construct detected during aggregation")
        grouped[(label.speaker_id, label.item_id)].append(label)

    units: List[UmeJrfCriterionUnit] = []
    for (speaker_id, item_id), rows in sorted(grouped.items()):
        rater_ids = [row.rater_id for row in rows]
        if len(rater_ids) != len(set(rater_ids)):
            raise ValueError(f"duplicate rater label for speaker={speaker_id} item={item_id}")
        raw = [int(row.raw_label) for row in rows]
        descriptions = {row.target_description for row in rows if row.target_description is not None}
        if len(descriptions) > 1:
            raise ValueError(
                f"conflicting target descriptions for speaker={speaker_id} item={item_id}"
            )
        description = next(iter(descriptions)) if descriptions else None

        ordinal_median = None
        binary_fraction = None
        binary_majority = None
        if scale == "ordinal_1_5":
            ordinal_median = float(statistics.median(raw))
        elif scale == "binary_correctness":
            binary_fraction = float(sum(raw) / len(raw))
            positives = sum(raw)
            negatives = len(raw) - positives
            binary_majority = 1 if positives > negatives else 0 if negatives > positives else None
        else:
            raise ValueError(f"unsupported criterion scale: {scale}")

        units.append(
            UmeJrfCriterionUnit(
                set_id=set_id,
                construct=construct,
                scale=scale,
                speaker_id=speaker_id,
                item_id=item_id,
                rater_ids=rater_ids,
                raw_labels=raw,
                rater_count=len(raw),
                ordinal_median=ordinal_median,
                binary_correct_fraction=binary_fraction,
                binary_majority_label=binary_majority,
                target_description=description,
                normalized_100=None,
                product_score_mapped=False,
                research_only_license=True,
                commercial_product_use_allowed=False,
            )
        )
    return units


def criterion_coverage_report(
    units: Sequence[UmeJrfCriterionUnit],
) -> Dict[str, Any]:
    """Report criterion coverage before any model-fitting decision."""
    if not units:
        return {
            "available": False,
            "reason": "no_criterion_units",
            "fit_allowed": False,
            "product_score_mapping_allowed": False,
        }
    sets = {unit.set_id for unit in units}
    constructs = {unit.construct for unit in units}
    scales = {unit.scale for unit in units}
    if len(sets) != 1 or len(constructs) != 1 or len(scales) != 1:
        raise ValueError("coverage report refuses mixed UME-JRF constructs")

    rater_counts = [unit.rater_count for unit in units]
    speakers = sorted({unit.speaker_id for unit in units})
    items = sorted({unit.item_id for unit in units})
    tied_binary = sum(
        unit.scale == "binary_correctness" and unit.binary_majority_label is None
        for unit in units
    )
    return {
        "available": True,
        "set_id": next(iter(sets)),
        "construct": next(iter(constructs)),
        "scale": next(iter(scales)),
        "unit_count": len(units),
        "speaker_count": len(speakers),
        "item_count": len(items),
        "rater_count_per_unit_min": min(rater_counts),
        "rater_count_per_unit_median": float(statistics.median(rater_counts)),
        "rater_count_per_unit_max": max(rater_counts),
        "binary_tied_majority_units": int(tied_binary),
        "raw_rater_labels_preserved": True,
        "normalized_100": None,
        "product_score_mapping_allowed": False,
        "commercial_product_training_allowed_without_separate_permission": False,
        "fit_allowed": len(speakers) >= 2 and len(items) >= 2,
    }


def make_group_holdout_folds(
    units: Sequence[UmeJrfCriterionUnit],
    *,
    group_by: Literal["speaker_id", "item_id"],
) -> List[Dict[str, Any]]:
    """Create deterministic leave-one-speaker/item-out fold indices.

    Returned folds contain only integer indices and pseudonymous group IDs.
    They do not copy audio or labels and do not decide that a criterion is good
    enough for product use.
    """
    if group_by not in {"speaker_id", "item_id"}:
        raise ValueError("group_by must be speaker_id or item_id")
    if not units:
        return []
    sets = {unit.set_id for unit in units}
    constructs = {unit.construct for unit in units}
    scales = {unit.scale for unit in units}
    if len(sets) != 1 or len(constructs) != 1 or len(scales) != 1:
        raise ValueError("holdout folds refuse mixed UME-JRF constructs")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        groups[str(getattr(unit, group_by))].append(index)
    if len(groups) < 2:
        raise ValueError(f"{group_by} holdout requires at least two unique groups")

    all_indices = set(range(len(units)))
    folds = []
    for group_id in sorted(groups):
        test_indices = sorted(groups[group_id])
        train_indices = sorted(all_indices - set(test_indices))
        folds.append(
            {
                "group_by": group_by,
                "held_out_group_id": group_id,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "train_unit_count": len(train_indices),
                "test_unit_count": len(test_indices),
                "speaker_or_item_ids_are_pseudonymous": True,
                "product_score_mapping_allowed": False,
            }
        )
    return folds


def build_validation_plan(units: Sequence[UmeJrfCriterionUnit]) -> Dict[str, Any]:
    """Freeze both speaker-held-out and item-held-out evaluation structure."""
    coverage = criterion_coverage_report(units)
    if not coverage.get("available"):
        return {
            "available": False,
            "coverage": coverage,
            "speaker_folds": [],
            "item_folds": [],
            "product_score_mapping_allowed": False,
        }
    if not coverage.get("fit_allowed"):
        return {
            "available": False,
            "reason": "insufficient_speaker_or_item_coverage",
            "coverage": coverage,
            "speaker_folds": [],
            "item_folds": [],
            "product_score_mapping_allowed": False,
        }
    return {
        "available": True,
        "coverage": coverage,
        "speaker_folds": make_group_holdout_folds(units, group_by="speaker_id"),
        "item_folds": make_group_holdout_folds(units, group_by="item_id"),
        "evaluation_policy": "report_both_speaker_held_out_and_item_held_out_before_phone_model_promotion",
        "raw_rater_labels_must_remain_available": True,
        "product_score_mapping_allowed": False,
        "commercial_product_training_allowed_without_separate_permission": False,
    }

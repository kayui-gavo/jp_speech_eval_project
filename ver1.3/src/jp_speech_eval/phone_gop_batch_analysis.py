"""Automatic analysis for the future manual Japanese phone-GOP battery.

The analyzer is intentionally useful *before* any human labels exist. It reads
manifest metadata plus per-clip GOP JSON artifacts and computes within-speaker
normal-repeat stability, intended-error deltas, focus localization, competitor
behavior and neighbor leakage. Optional blind-listening labels can be merged
later without changing the acoustic analysis schema.

This module never maps GOP to a user-facing /100 score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


METRICS = (
    "mean_logit_margin",
    "max_logit_margin",
    "posterior_gop_margin",
    "mean_entropy",
)


@dataclass(frozen=True)
class FocusMetric:
    phone_index: int
    phone: str
    mean_logit_margin: float
    max_logit_margin: float
    posterior_gop_margin: float
    mean_entropy: float
    best_competitor_phone: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _gop_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    gop = payload.get("gop")
    if isinstance(gop, Mapping):
        return gop
    return payload


def _phone_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    gop = _gop_payload(payload)
    rows = gop.get("evidence")
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def resolve_focus_indices(rows: Sequence[Mapping[str, Any]], focus: str) -> list[int]:
    """Resolve manifest focus notation against canonical GOP phone rows."""
    token = str(focus or "").strip()
    phones = [str(row.get("canonical_phone") or "") for row in rows]
    if not token or token == "whole":
        return list(range(len(rows)))
    if token == "high_vowels":
        return [index for index, phone in enumerate(phones) if phone in {"i", "I", "u", "U"}]
    if token.startswith("second_"):
        target = token[len("second_") :]
        matches = [index for index, phone in enumerate(phones) if phone == target]
        return matches[1:2]
    if token.startswith("first_"):
        target = token[len("first_") :]
        matches = [index for index, phone in enumerate(phones) if phone == target]
        return matches[:1]
    return [index for index, phone in enumerate(phones) if phone == token]


def focus_metrics(payload: Mapping[str, Any], focus: str) -> list[FocusMetric]:
    rows = _phone_rows(payload)
    indices = resolve_focus_indices(rows, focus)
    output: list[FocusMetric] = []
    for index in indices:
        row = rows[index]
        values = {metric: _finite(row.get(metric)) for metric in METRICS}
        if any(values[metric] is None for metric in METRICS):
            continue
        output.append(FocusMetric(
            phone_index=int(row.get("phone_index", index)),
            phone=str(row.get("canonical_phone") or ""),
            mean_logit_margin=float(values["mean_logit_margin"]),
            max_logit_margin=float(values["max_logit_margin"]),
            posterior_gop_margin=float(values["posterior_gop_margin"]),
            mean_entropy=float(values["mean_entropy"]),
            best_competitor_phone=str(row.get("best_competitor_phone") or ""),
        ))
    return output


def _aggregate(rows: Sequence[FocusMetric]) -> Dict[str, Any]:
    if not rows:
        return {"available": False}
    result: Dict[str, Any] = {"available": True, "focus_phone_count": len(rows)}
    for metric in METRICS:
        values = [float(getattr(row, metric)) for row in rows]
        result[metric] = sum(values) / len(values)
    result["focus_phones"] = [row.phone for row in rows]
    result["focus_indices"] = [row.phone_index for row in rows]
    result["best_competitors"] = [row.best_competitor_phone for row in rows]
    return result


def _absolute_delta(a: Mapping[str, Any], b: Mapping[str, Any], metric: str) -> Optional[float]:
    x = _finite(a.get(metric))
    y = _finite(b.get(metric))
    if x is None or y is None:
        return None
    return abs(x - y)


def _signed_delta(error: Mapping[str, Any], baseline: Mapping[str, Any], metric: str) -> Optional[float]:
    e = _finite(error.get(metric))
    b = _finite(baseline.get(metric))
    if e is None or b is None:
        return None
    return e - b


def _mean_dict(dicts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    usable = [row for row in dicts if row.get("available")]
    if not usable:
        return {"available": False}
    out: Dict[str, Any] = {"available": True}
    for metric in METRICS:
        values = [_finite(row.get(metric)) for row in usable]
        finite = [value for value in values if value is not None]
        out[metric] = sum(finite) / len(finite) if finite else None
    return out


def _neighbor_delta(
    baseline_payload: Mapping[str, Any],
    error_payload: Mapping[str, Any],
    focus_indices: Sequence[int],
    metric: str = "posterior_gop_margin",
) -> Optional[float]:
    baseline_rows = _phone_rows(baseline_payload)
    error_rows = _phone_rows(error_payload)
    if len(baseline_rows) != len(error_rows) or not focus_indices:
        return None
    neighbors: set[int] = set()
    for index in focus_indices:
        if index - 1 >= 0:
            neighbors.add(index - 1)
        if index + 1 < len(baseline_rows):
            neighbors.add(index + 1)
    deltas: list[float] = []
    for index in sorted(neighbors):
        a = _finite(baseline_rows[index].get(metric))
        b = _finite(error_rows[index].get(metric))
        if a is not None and b is not None:
            deltas.append(abs(b - a))
    return max(deltas) if deltas else None


def _expected_competitor_match(expected: str, observed: Sequence[str]) -> Optional[bool]:
    token = str(expected or "").strip()
    if not token or token in {"unknown", "blank_or_neighbor", "a_or_blank", "i_y", "long_vowel"}:
        return None
    return token in {str(item) for item in observed if str(item)}


def analyze_manual_gop_batch(
    manifest_rows: Sequence[Mapping[str, str]],
    clip_payloads: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Analyze all available N1/N2/E groups without requiring human labels."""
    groups: Dict[str, list[Mapping[str, str]]] = {}
    for row in manifest_rows:
        group = str(row.get("repeat_group") or row.get("clip_id") or "").strip()
        if group:
            groups.setdefault(group, []).append(row)

    analyses: list[Dict[str, Any]] = []
    missing_clips: list[str] = []
    for group, rows in sorted(groups.items()):
        present = [row for row in rows if str(row.get("clip_id")) in clip_payloads]
        for row in rows:
            clip_id = str(row.get("clip_id") or "")
            if clip_id and clip_id not in clip_payloads:
                missing_clips.append(clip_id)
        if not present:
            continue

        clip_metrics: Dict[str, Dict[str, Any]] = {}
        row_by_clip: Dict[str, Mapping[str, str]] = {}
        for row in present:
            clip_id = str(row.get("clip_id") or "")
            row_by_clip[clip_id] = row
            focus = str(row.get("target_focus") or "")
            clip_metrics[clip_id] = _aggregate(focus_metrics(clip_payloads[clip_id], focus))

        normal_ids = [
            clip_id for clip_id, row in row_by_clip.items()
            if str(row.get("condition") or "").startswith("normal")
        ]
        error_ids = [
            clip_id for clip_id, row in row_by_clip.items()
            if str(row.get("condition") or "") == "error"
        ]
        normal_metrics = [clip_metrics[clip_id] for clip_id in normal_ids]
        baseline = _mean_dict(normal_metrics)

        clean_repeat_delta: Dict[str, Optional[float]] = {}
        if len(normal_ids) >= 2:
            first, second = clip_metrics[normal_ids[0]], clip_metrics[normal_ids[1]]
            clean_repeat_delta = {
                metric: _absolute_delta(first, second, metric) for metric in METRICS
            }

        error_summaries: list[Dict[str, Any]] = []
        for clip_id in error_ids:
            error_metric = clip_metrics[clip_id]
            row = row_by_clip[clip_id]
            focus_indices = error_metric.get("focus_indices", []) if error_metric.get("available") else []
            observed_competitors = error_metric.get("best_competitors", []) if error_metric.get("available") else []
            signed = {
                metric: _signed_delta(error_metric, baseline, metric) for metric in METRICS
            }
            clean_scale = clean_repeat_delta.get("posterior_gop_margin")
            error_drop = signed.get("posterior_gop_margin")
            separation_ratio = None
            if clean_scale is not None and error_drop is not None:
                separation_ratio = abs(error_drop) / max(clean_scale, 1e-6)

            neighbor = None
            if normal_ids and focus_indices:
                neighbor = _neighbor_delta(
                    clip_payloads[normal_ids[0]],
                    clip_payloads[clip_id],
                    focus_indices,
                )
            error_summaries.append({
                "clip_id": clip_id,
                "focus": str(row.get("target_focus") or ""),
                "intended_error_type": str(row.get("target_error_type") or ""),
                "expected_competitor": str(row.get("expected_competitor") or ""),
                "focus_metrics": error_metric,
                "error_minus_clean_baseline": signed,
                "posterior_error_to_clean_repeat_ratio": separation_ratio,
                "neighbor_posterior_delta_max": neighbor,
                "expected_competitor_match": _expected_competitor_match(
                    str(row.get("expected_competitor") or ""), observed_competitors
                ),
            })

        analyses.append({
            "repeat_group": group,
            "component": str(rows[0].get("component") or ""),
            "target_text": str(rows[0].get("target_text") or ""),
            "target_focus": str(rows[0].get("target_focus") or ""),
            "normal_clip_ids": normal_ids,
            "error_clip_ids": error_ids,
            "clean_baseline": baseline,
            "clean_repeat_absolute_delta": clean_repeat_delta,
            "errors": error_summaries,
        })

    complete_error_groups = [row for row in analyses if row["normal_clip_ids"] and row["error_clip_ids"]]
    strong_direction = 0
    evaluated_direction = 0
    for group in complete_error_groups:
        for error in group["errors"]:
            delta = error["error_minus_clean_baseline"].get("posterior_gop_margin")
            if delta is not None:
                evaluated_direction += 1
                if delta < 0:
                    strong_direction += 1

    return {
        "schema": "phone_gop_manual_batch_analysis_v1",
        "score_mapped": False,
        "product_calibrated": False,
        "human_recording_gate_changed": False,
        "groups_analyzed": len(analyses),
        "complete_error_groups": len(complete_error_groups),
        "missing_clip_count": len(set(missing_clips)),
        "missing_clips": sorted(set(missing_clips)),
        "directional_error_groups_evaluated": evaluated_direction,
        "directional_error_groups_with_lower_posterior_margin": strong_direction,
        "groups": analyses,
    }

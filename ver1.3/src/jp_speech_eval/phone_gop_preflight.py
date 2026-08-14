from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from .japanese_phoneme_gop import NON_SEGMENTAL_TOKENS, JapanesePhoneCtcBackend
from .japanese_target_evidence import JapaneseTargetEvidence
from .phoneme_gop import PhoneGopResult


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str  # pass | warn | block
    detail: str
    metrics: Dict[str, Any]
    required_for_human_recording: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhoneGopPreflightReport:
    checks: list[PreflightCheck]
    artifacts: Dict[str, Any]

    @property
    def human_recording_allowed(self) -> bool:
        return not any(
            check.required_for_human_recording and check.status == "block"
            for check in self.checks
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "japanese_phone_gop_preflight_v1",
            "human_recording_allowed": self.human_recording_allowed,
            "checks": [check.to_dict() for check in self.checks],
            "artifacts": self.artifacts,
        }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _summary_number(result: PhoneGopResult, key: str) -> float | None:
    value = result.summary.get(key)
    if _finite(value):
        return float(value)
    return None


def _result_has_only_finite_numeric_evidence(result: PhoneGopResult) -> bool:
    if not result.available:
        return False
    for row in result.evidence:
        for key, value in row.to_dict().items():
            if isinstance(value, (int, float, np.floating, np.integer)) and not _finite(value):
                return False
    for key in ("ctc_forward_logprob", "ctc_forward_logprob_per_frame"):
        value = result.summary.get(key)
        if value is not None and not _finite(value):
            return False
    return True


def _best_competitors(result: PhoneGopResult) -> set[str]:
    return {str(row.best_competitor_phone) for row in result.evidence}


def _check_frontend(target: JapaneseTargetEvidence) -> PreflightCheck:
    if target.frontend_ambiguous:
        return PreflightCheck(
            name="target_frontend_distribution",
            status="block",
            detail="Both pyopenjtalk and pyopenjtalk-plus are installed; the import owner is ambiguous.",
            metrics={
                "distribution": target.frontend_distribution,
                "version": target.frontend_version,
                "warnings": target.warnings,
            },
        )
    if target.frontend_distribution != "pyopenjtalk-plus":
        return PreflightCheck(
            name="target_frontend_distribution",
            status="block",
            detail="Selected Japanese phone-CTC model was trained with pyopenjtalk-plus labels, but the active target frontend is not pyopenjtalk-plus.",
            metrics={
                "distribution": target.frontend_distribution,
                "version": target.frontend_version,
                "warnings": target.warnings,
            },
        )
    return PreflightCheck(
        name="target_frontend_distribution",
        status="pass",
        detail="Active target frontend is pyopenjtalk-plus, matching the selected backend training-label family.",
        metrics={"distribution": target.frontend_distribution, "version": target.frontend_version},
    )


def _check_target_inventory(
    backend: JapanesePhoneCtcBackend,
    targets: Mapping[str, JapaneseTargetEvidence],
) -> PreflightCheck:
    try:
        raw_vocab = backend.vocabulary()
    except Exception as exc:
        return PreflightCheck(
            name="target_phone_inventory_coverage",
            status="block",
            detail=f"Could not inspect backend vocabulary: {type(exc).__name__}: {exc}",
            metrics={},
        )

    # The backend itself projects i/I and u/U into logical classes. For static
    # coverage, accept either allophone spelling and ignore text-level pauses.
    logical_tokens = set(raw_vocab)
    if "I" in logical_tokens or "i" in logical_tokens:
        logical_tokens.update({"i", "I"})
    if "U" in logical_tokens or "u" in logical_tokens:
        logical_tokens.update({"u", "U"})

    missing: Dict[str, list[str]] = {}
    for label, target in targets.items():
        unsupported = sorted({
            str(phone)
            for phone in target.phones
            if str(phone) not in {"pau", "sil"} and str(phone) not in logical_tokens
        })
        if unsupported:
            missing[label] = unsupported

    if missing:
        return PreflightCheck(
            name="target_phone_inventory_coverage",
            status="block",
            detail="At least one preflight target contains phones not supported by the selected backend vocabulary.",
            metrics={"missing_by_target": missing, "vocab_size": len(raw_vocab)},
        )
    return PreflightCheck(
        name="target_phone_inventory_coverage",
        status="pass",
        detail="All preflight target phones are covered by the selected Japanese phone inventory.",
        metrics={"target_count": len(targets), "vocab_size": len(raw_vocab)},
    )


def _check_result_contract(label: str, result: PhoneGopResult) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    if not result.available:
        checks.append(PreflightCheck(
            name=f"{label}_gop_available",
            status="block",
            detail="Phone-GOP backend did not return usable evidence.",
            metrics={"summary": result.summary, "warnings": result.warnings},
        ))
        return checks

    checks.append(PreflightCheck(
        name=f"{label}_gop_available",
        status="pass",
        detail="Phone-GOP backend returned evidence.",
        metrics={
            "supported_phone_count": result.summary.get("supported_phone_count"),
            "model_id": result.model_id,
            "method": result.method,
        },
    ))

    finite = _result_has_only_finite_numeric_evidence(result)
    checks.append(PreflightCheck(
        name=f"{label}_finite_evidence",
        status="pass" if finite else "block",
        detail="All numeric phone evidence is finite." if finite else "Non-finite phone evidence was found.",
        metrics={},
    ))

    competitors = _best_competitors(result)
    illegal = sorted(competitors & set(NON_SEGMENTAL_TOKENS))
    checks.append(PreflightCheck(
        name=f"{label}_segmental_competitors_only",
        status="pass" if not illegal else "block",
        detail=(
            "No pause/silence/special token is used as a segmental competitor."
            if not illegal else
            "Non-segmental tokens leaked into phone competition."
        ),
        metrics={"illegal_competitors": illegal},
    ))
    return checks


def _check_wrong_target_separation(correct: PhoneGopResult, wrong: PhoneGopResult) -> PreflightCheck:
    correct_lp = _summary_number(correct, "ctc_forward_logprob_per_frame")
    wrong_lp = _summary_number(wrong, "ctc_forward_logprob_per_frame")
    if correct_lp is None or wrong_lp is None:
        return PreflightCheck(
            name="correct_vs_wrong_target_separation",
            status="block",
            detail="Sequence-level CTC likelihood is missing for correct/wrong target comparison.",
            metrics={"correct": correct_lp, "wrong": wrong_lp},
        )

    gap = correct_lp - wrong_lp
    correct_ed = correct.summary.get("greedy_phone_edit_distance")
    wrong_ed = wrong.summary.get("greedy_phone_edit_distance")
    edit_direction_ok = (
        isinstance(correct_ed, int) and isinstance(wrong_ed, int) and correct_ed <= wrong_ed
    )
    # The first hard gate is deliberately modest; the eventual product threshold
    # must come from distributions, not this single TTS reference. A non-positive
    # gap means the backend cannot even rank the known target above a deliberately
    # wrong target on bundled audio and should not consume human time.
    passed = gap > 0.01 and edit_direction_ok
    return PreflightCheck(
        name="correct_vs_wrong_target_separation",
        status="pass" if passed else "block",
        detail=(
            "Bundled reference is better explained by the correct target than the wrong target."
            if passed else
            "Correct-target evidence is not reliably better than wrong-target evidence on bundled audio."
        ),
        metrics={
            "correct_logprob_per_frame": correct_lp,
            "wrong_logprob_per_frame": wrong_lp,
            "gap": gap,
            "correct_greedy_edit_distance": correct_ed,
            "wrong_greedy_edit_distance": wrong_ed,
        },
    )


def _check_gain_stability(
    clean: PhoneGopResult,
    gain_results: Mapping[str, PhoneGopResult],
    *,
    wrong_target_gap: float | None,
) -> PreflightCheck:
    clean_lp = _summary_number(clean, "ctc_forward_logprob_per_frame")
    if clean_lp is None:
        return PreflightCheck(
            name="gain_stability",
            status="block",
            detail="Clean sequence likelihood is unavailable.",
            metrics={},
        )
    deltas: Dict[str, float] = {}
    for label, result in gain_results.items():
        value = _summary_number(result, "ctc_forward_logprob_per_frame")
        if value is None:
            return PreflightCheck(
                name="gain_stability",
                status="block",
                detail=f"Gain perturbation {label} has no sequence likelihood.",
                metrics={},
            )
        deltas[label] = abs(value - clean_lp)

    max_delta = max(deltas.values(), default=0.0)
    # The perturbation should be much less damaging than replacing the target.
    # Use a small absolute allowance because the selected feature extractor does
    # not normalize amplitude. This remains a preflight engineering gate, not a
    # scientific pronunciation threshold.
    if wrong_target_gap is not None and wrong_target_gap > 0:
        threshold = max(0.10, 0.60 * wrong_target_gap)
    else:
        threshold = 0.10
    passed = max_delta <= threshold
    return PreflightCheck(
        name="gain_stability",
        status="pass" if passed else "block",
        detail=(
            "Mild gain changes perturb sequence evidence less than the configured preflight allowance."
            if passed else
            "Phone evidence is too sensitive to mild gain changes relative to target separation."
        ),
        metrics={"absolute_deltas": deltas, "max_delta": max_delta, "threshold": threshold},
    )


def build_phone_gop_preflight_report(
    *,
    backend: JapanesePhoneCtcBackend,
    correct_target: JapaneseTargetEvidence,
    wrong_target: JapaneseTargetEvidence,
    extra_targets: Mapping[str, JapaneseTargetEvidence],
    correct_result: PhoneGopResult,
    wrong_result: PhoneGopResult,
    gain_results: Mapping[str, PhoneGopResult],
) -> PhoneGopPreflightReport:
    targets = {"correct": correct_target, "wrong": wrong_target, **dict(extra_targets)}
    checks: list[PreflightCheck] = [_check_frontend(correct_target)]
    checks.append(_check_target_inventory(backend, targets))
    checks.extend(_check_result_contract("correct", correct_result))
    checks.extend(_check_result_contract("wrong", wrong_result))
    for label, result in gain_results.items():
        checks.extend(_check_result_contract(label, result))

    separation = _check_wrong_target_separation(correct_result, wrong_result)
    checks.append(separation)
    gap = separation.metrics.get("gap")
    checks.append(_check_gain_stability(
        correct_result,
        gain_results,
        wrong_target_gap=float(gap) if _finite(gap) else None,
    ))

    flags = correct_result.summary
    checks.append(PreflightCheck(
        name="japanese_allophone_policy_active",
        status="pass" if bool(flags.get("high_vowel_allophones_collapsed")) else "block",
        detail="i/I and u/U are collapsed for clarity scoring." if bool(flags.get("high_vowel_allophones_collapsed")) else "Japanese high-vowel allophone policy is not active.",
        metrics={},
    ))
    checks.append(PreflightCheck(
        name="ctc_support_not_phone_duration",
        status="pass" if bool(flags.get("ctc_support_frames_are_not_physical_phone_boundaries")) else "block",
        detail="CTC support frames are explicitly kept separate from physical phone duration." if bool(flags.get("ctc_support_frames_are_not_physical_phone_boundaries")) else "CTC support duration semantics are not explicit.",
        metrics={},
    ))
    checks.append(PreflightCheck(
        name="no_product_score_mapping",
        status="pass" if (not correct_result.score_mapped and not correct_result.product_calibrated) else "block",
        detail="Phone evidence remains shadow-only and unmapped to /100." if (not correct_result.score_mapped and not correct_result.product_calibrated) else "Phone evidence has been mapped/promoted prematurely.",
        metrics={"score_mapped": correct_result.score_mapped, "product_calibrated": correct_result.product_calibrated},
    ))

    return PhoneGopPreflightReport(
        checks=checks,
        artifacts={
            "correct_target": correct_target.to_dict(),
            "wrong_target": wrong_target.to_dict(),
            "extra_target_count": len(extra_targets),
            "correct_result": correct_result.to_dict(),
            "wrong_result": wrong_result.to_dict(),
            "gain_results": {label: result.to_dict() for label, result in gain_results.items()},
        },
    )

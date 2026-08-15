from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from jp_speech_eval.score_contract import history_comparability


STEP_SHADOWING = 1
STEP_FADED_REFERENCE = 2
STEP_FREE_PRODUCTION = 3


@dataclass
class ReferenceDependencyGap:
    item_id: str
    shadowing_score: Optional[float]
    free_production_score: Optional[float]
    gap: Optional[float]
    feedback: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def step_label(step: int) -> str:
    labels = {
        STEP_SHADOWING: "Step 1 听参考音跟读",
        STEP_FADED_REFERENCE: "Step 2 看提示自己读",
        STEP_FREE_PRODUCTION: "Step 3 不听参考音表达",
    }
    return labels.get(int(step), f"Step {step}")


def step_user_instruction(step: int) -> str:
    instructions = {
        STEP_SHADOWING: "先听参考音，再模仿语速、停顿和句末语调。",
        STEP_FADED_REFERENCE: "这次不播放完整参考音，只看假名或音高提示来读。",
        STEP_FREE_PRODUCTION: "不听参考音，按场景或目标意思自己说出来。",
    }
    return instructions.get(int(step), "按当前练习要求录音。")


def _score(record: Optional[Dict[str, Any]]) -> Optional[float]:
    if not record:
        return None
    try:
        return float((record.get("scores") or {}).get("total"))
    except (TypeError, ValueError):
        return None


def _feature(record: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not record:
        return None
    try:
        value = (record.get("features") or {}).get(key)
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _pct_change(first: Optional[float], second: Optional[float]) -> Optional[float]:
    if first is None or second is None or abs(first) < 1e-8:
        return None
    return round((second - first) / first * 100.0, 2)


def compute_reference_dependency_gap(
    records: Iterable[Dict[str, Any]],
    *,
    item_id: str,
) -> ReferenceDependencyGap:
    """Describe Step 1 -> Step 3 change without inventing a cross-mode score gap.

    A shadowing trial and a free-production trial often use different evidence
    families. Their headline /100 scores therefore cannot automatically be
    subtracted. Observable features such as speech rate and pause ratio remain
    useful descriptive cross-step signals.
    """

    step1 = None
    step3 = None
    for record in records:
        if record.get("item_id") != item_id:
            continue
        if int(record.get("step", -1)) == STEP_SHADOWING:
            step1 = record
        elif int(record.get("step", -1)) == STEP_FREE_PRODUCTION:
            step3 = record

    shadowing_score = _score(step1)
    free_score = _score(step3)
    if step1 is None or step3 is None:
        return ReferenceDependencyGap(
            item_id=item_id,
            shadowing_score=shadowing_score,
            free_production_score=free_score,
            gap=None,
            feedback=["完成 Step 1 和 Step 3 后，可以比较不依赖评分规则的语速和停顿变化。"],
            debug={"step1_recorded": step1 is not None, "step3_recorded": step3 is not None},
        )

    comparability = history_comparability(
        step1.get("score_context") or {},
        step3.get("score_context") or {},
    )
    rate1 = _feature(step1, "mora_rate")
    rate3 = _feature(step3, "mora_rate")
    pause1 = _feature(step1, "pause_ratio")
    pause3 = _feature(step3, "pause_ratio")
    feature_change = {
        "mora_rate_pct_step1_to_step3": _pct_change(rate1, rate3),
        "pause_ratio_delta_step1_to_step3": (
            None if pause1 is None or pause3 is None else round(pause3 - pause1, 4)
        ),
    }

    if not comparability["comparable"]:
        feedback = ["跟读和自由表达使用的评分依据不同，因此不直接用总分差判断是否依赖参考音。"]
        if feature_change["mora_rate_pct_step1_to_step3"] is not None:
            rate_change = float(feature_change["mora_rate_pct_step1_to_step3"])
            if rate_change <= -20:
                feedback.append("不听参考音后语速明显下降，可以先减少参考音次数，再逐步过渡到自己说。")
        if feature_change["pause_ratio_delta_step1_to_step3"] is not None:
            pause_delta = float(feature_change["pause_ratio_delta_step1_to_step3"])
            if pause_delta >= 0.12:
                feedback.append("不听参考音后停顿明显增加，说明独立组织表达还需要练习。")
        return ReferenceDependencyGap(
            item_id=item_id,
            shadowing_score=shadowing_score,
            free_production_score=free_score,
            gap=None,
            feedback=feedback,
            debug={
                "step1_recorded": True,
                "step3_recorded": True,
                "score_comparability": comparability,
                "feature_change": feature_change,
            },
        )

    if shadowing_score is None or free_score is None:
        return ReferenceDependencyGap(
            item_id=item_id,
            shadowing_score=shadowing_score,
            free_production_score=free_score,
            gap=None,
            feedback=["两次练习的评分规则一致，但至少一次没有可用总分。"],
            debug={"score_comparability": comparability, "feature_change": feature_change},
        )

    gap = round(shadowing_score - free_score, 2)
    if gap >= 12:
        feedback = ["在可比较的同一评分条件下，这次独立表达比跟读低很多，可以逐步减少参考音。"]
    elif gap <= 5:
        feedback = ["在可比较的同一评分条件下，不听参考音后整体表现保持得比较稳定。"]
    else:
        feedback = ["在可比较的同一评分条件下，你正在从跟读过渡到自己说。"]
    return ReferenceDependencyGap(
        item_id=item_id,
        shadowing_score=shadowing_score,
        free_production_score=free_score,
        gap=gap,
        feedback=feedback,
        debug={
            "step1_recorded": True,
            "step3_recorded": True,
            "score_comparability": comparability,
            "feature_change": feature_change,
        },
    )

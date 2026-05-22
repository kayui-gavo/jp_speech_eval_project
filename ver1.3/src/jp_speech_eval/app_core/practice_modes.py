from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


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


def compute_reference_dependency_gap(
    records: Iterable[Dict[str, Any]],
    *,
    item_id: str,
) -> ReferenceDependencyGap:
    """Compare Step 1 and Step 3 to detect shadowing dependence."""

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
    if shadowing_score is None or free_score is None:
        return ReferenceDependencyGap(
            item_id=item_id,
            shadowing_score=shadowing_score,
            free_production_score=free_score,
            gap=None,
            feedback=["完成 Step 1 和 Step 3 后，可以看到你是否太依赖参考音。"],
        )

    gap = round(shadowing_score - free_score, 2)
    if gap >= 12:
        feedback = ["你跟读时表现很好，但自己说时还容易掉语调。下一次可以多看提示少听参考音。"]
    elif gap <= 5:
        feedback = ["你已经能在没有参考音时保持比较自然，可以进入更自由的场景练习。"]
    else:
        feedback = ["你正在从跟读过渡到自己说。下一步建议减少播放参考音的次数。"]
    return ReferenceDependencyGap(
        item_id=item_id,
        shadowing_score=shadowing_score,
        free_production_score=free_score,
        gap=gap,
        feedback=feedback,
        debug={"step1_recorded": step1 is not None, "step3_recorded": step3 is not None},
    )

"""Speaker-held-out criterion validation for phone-level research features.

This module is intentionally independent of the consumer score.  Once an
expert-labeled Japanese learner corpus is available, it evaluates whether phone
features actually discriminate expert-correct from expert-incorrect
realizations without speaker leakage.

Design principles:

* groups are speakers; no utterance-level random split is allowed;
* labels are explicit binary *criterion* labels, not generated from GOP itself;
* all preprocessing/model fitting happens inside each training fold;
* out-of-fold probabilities are retained for phone-specific false-alarm and
  miss analysis;
* no threshold is promoted to the C-end score by this module;
* model comparison should be based on held-out criterion performance, not raw
  GOP magnitude.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


SCHEMA = "speaker_held_out_phone_criterion_validation_v1"


@dataclass(frozen=True)
class PhoneCriterionExample:
    sample_id: str
    speaker_id: str
    canonical_phone: str
    is_expert_correct: bool
    features: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finite_feature_matrix(
    examples: Sequence[PhoneCriterionExample],
    feature_names: Sequence[str],
) -> np.ndarray:
    matrix = np.asarray(
        [[float(example.features[name]) for name in feature_names] for example in examples],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[0] != len(examples):
        raise ValueError("invalid criterion feature matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("criterion feature matrix contains non-finite values")
    return matrix


def _choose_splits(labels: np.ndarray, groups: np.ndarray, requested: int) -> int:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("speaker-held-out validation requires at least two speakers")
    # StratifiedGroupKFold also needs each class to be represented across enough
    # speakers. Use the most conservative class-specific speaker count.
    class_group_counts = []
    for label in np.unique(labels):
        class_group_counts.append(np.unique(groups[labels == label]).size)
    if len(class_group_counts) < 2 or min(class_group_counts) < 2:
        raise ValueError("each criterion class must occur in at least two speakers")
    return max(2, min(int(requested), int(unique_groups.size), int(min(class_group_counts))))


def _eer(labels: np.ndarray, error_probability: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, error_probability)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def evaluate_phone_criterion_binary(
    examples: Sequence[PhoneCriterionExample],
    *,
    feature_names: Sequence[str],
    n_splits: int = 5,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Run out-of-fold speaker-held-out logistic criterion validation.

    The positive class for model fitting is **expert incorrect** so reported ROC
    AUC describes error detection.  `0.5` decisions below are diagnostic only;
    no learner-facing threshold is created.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rows = list(examples)
    if len(rows) < 4:
        raise ValueError("criterion validation requires at least four examples")
    names = [str(name) for name in feature_names]
    if not names:
        raise ValueError("feature_names must not be empty")
    for example in rows:
        missing = [name for name in names if name not in example.features]
        if missing:
            raise ValueError(f"missing criterion features for {example.sample_id}: {missing}")

    x = _finite_feature_matrix(rows, names)
    # Error detection label: 1 = expert says incorrect, 0 = expert says correct.
    y = np.asarray([0 if example.is_expert_correct else 1 for example in rows], dtype=np.int64)
    groups = np.asarray([str(example.speaker_id) for example in rows], dtype=object)
    phones = np.asarray([str(example.canonical_phone) for example in rows], dtype=object)
    sample_ids = [str(example.sample_id) for example in rows]
    if np.unique(y).size != 2:
        raise ValueError("criterion labels must contain both correct and incorrect examples")

    splits = _choose_splits(y, groups, n_splits)
    splitter = StratifiedGroupKFold(
        n_splits=splits,
        shuffle=True,
        random_state=int(random_state),
    )
    oof_probability = np.full(len(rows), np.nan, dtype=np.float64)
    fold_rows: List[Dict[str, Any]] = []

    for fold, (train, test) in enumerate(splitter.split(x, y, groups=groups)):
        if set(groups[train]).intersection(set(groups[test])):
            raise RuntimeError("speaker leakage detected in criterion split")
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=int(random_state),
                    ),
                ),
            ]
        )
        pipeline.fit(x[train], y[train])
        probability = pipeline.predict_proba(x[test])[:, 1]
        oof_probability[test] = probability
        fold_rows.append(
            {
                "fold": fold,
                "train_speakers": sorted(set(groups[train].tolist())),
                "test_speakers": sorted(set(groups[test].tolist())),
                "train_n": int(train.size),
                "test_n": int(test.size),
                "speaker_overlap": False,
            }
        )

    if not np.isfinite(oof_probability).all():
        raise RuntimeError("criterion splitter did not produce an OOF probability for every example")

    decisions = (oof_probability >= 0.5).astype(np.int64)
    auc = float(roc_auc_score(y, oof_probability))
    balanced = float(balanced_accuracy_score(y, decisions))
    eer = _eer(y, oof_probability)

    phone_rows: Dict[str, Dict[str, Any]] = {}
    for phone in sorted(set(phones.tolist())):
        mask = phones == phone
        correct_mask = mask & (y == 0)
        incorrect_mask = mask & (y == 1)
        phone_rows[phone] = {
            "n": int(np.sum(mask)),
            "expert_correct_n": int(np.sum(correct_mask)),
            "expert_incorrect_n": int(np.sum(incorrect_mask)),
            "diagnostic_false_alarm_rate_at_0_5": (
                float(np.mean(decisions[correct_mask] == 1)) if np.any(correct_mask) else None
            ),
            "diagnostic_miss_rate_at_0_5": (
                float(np.mean(decisions[incorrect_mask] == 0)) if np.any(incorrect_mask) else None
            ),
            "phone_threshold_calibrated": False,
        }

    predictions = [
        {
            "sample_id": sample_id,
            "speaker_id": str(groups[index]),
            "canonical_phone": str(phones[index]),
            "is_expert_correct": bool(y[index] == 0),
            "error_probability_oof": float(oof_probability[index]),
            "diagnostic_error_decision_at_0_5": bool(decisions[index]),
        }
        for index, sample_id in enumerate(sample_ids)
    ]

    return {
        "schema": SCHEMA,
        "available": True,
        "example_count": len(rows),
        "speaker_count": int(np.unique(groups).size),
        "feature_names": names,
        "n_splits": splits,
        "positive_class": "expert_incorrect",
        "split_policy": "StratifiedGroupKFold_by_speaker",
        "model": "StandardScaler+class_balanced_logistic_regression",
        "metrics": {
            "roc_auc_error_detection": auc,
            "balanced_accuracy_at_diagnostic_0_5": balanced,
            "eer": eer,
        },
        "folds": fold_rows,
        "per_phone": phone_rows,
        "predictions": predictions,
        "summary": {
            "speaker_leakage_allowed": False,
            "criterion_labels_generated_from_GOP": False,
            "diagnostic_0_5_is_product_threshold": False,
            "phone_specific_thresholds_calibrated": False,
            "score_mapped": False,
            "product_calibrated": False,
            "product_score_changed": False,
            "intended_use": "compare_phone_feature_families_on_expert_labeled_Japanese_L2_speech",
        },
    }

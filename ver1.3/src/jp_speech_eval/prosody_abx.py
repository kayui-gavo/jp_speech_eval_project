from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import librosa
import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class DtwAlignment:
    """DTW result with a per-X-frame local distance curve."""

    normalized_cost: float
    path_length: int
    x_distance_curve: List[float]


@dataclass(frozen=True)
class AbxTrialResult:
    """One Prosodic ABX decision."""

    d_ax: float
    d_bx: float
    margin: float
    correct: bool
    ax_curve: List[float]
    bx_curve: List[float]
    evidence_curve: List[float]

    def to_dict(self) -> Dict:
        return asdict(self)


def _as_2d(name: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError(f"{name} must have shape [T, D] with at least 2 frames; got {x.shape}")
    return x


def dtw_alignment(x: np.ndarray, y: np.ndarray, metric: str = "cosine") -> DtwAlignment:
    """Compute normalized DTW cost and local distances aggregated over Y frames."""

    x = _as_2d("x", x)
    y = _as_2d("y", y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"feature dimensions differ: {x.shape[1]} vs {y.shape[1]}")

    D, wp = librosa.sequence.dtw(X=x.T, Y=y.T, metric=metric)
    path = np.asarray(wp, dtype=int)
    if path.size == 0:
        raise ValueError("empty DTW path")

    local_matrix = cdist(x, y, metric=metric)
    local_costs = local_matrix[path[:, 0], path[:, 1]]
    normalized_cost = float(D[-1, -1] / max(len(path), 1))

    curve = np.full(y.shape[0], np.nan, dtype=float)
    for y_idx in np.unique(path[:, 1]):
        curve[y_idx] = float(np.mean(local_costs[path[:, 1] == y_idx]))
    valid = np.isfinite(curve)
    if not np.all(valid):
        valid_idx = np.where(valid)[0]
        curve = np.interp(np.arange(len(curve)), valid_idx, curve[valid_idx])
    return DtwAlignment(
        normalized_cost=normalized_cost,
        path_length=int(len(path)),
        x_distance_curve=[float(v) for v in curve],
    )


def score_abx(a: np.ndarray, b: np.ndarray, x: np.ndarray, metric: str = "cosine") -> AbxTrialResult:
    """Score one ABX trial where X is expected to match A."""

    ax = dtw_alignment(a, x, metric=metric)
    bx = dtw_alignment(b, x, metric=metric)
    n = min(len(ax.x_distance_curve), len(bx.x_distance_curve))
    ax_curve = np.asarray(ax.x_distance_curve[:n], dtype=float)
    bx_curve = np.asarray(bx.x_distance_curve[:n], dtype=float)
    evidence_curve = bx_curve - ax_curve
    margin = float(bx.normalized_cost - ax.normalized_cost)
    return AbxTrialResult(
        d_ax=round(ax.normalized_cost, 6),
        d_bx=round(bx.normalized_cost, 6),
        margin=round(margin, 6),
        correct=bool(ax.normalized_cost < bx.normalized_cost),
        ax_curve=[round(float(v), 6) for v in ax_curve],
        bx_curve=[round(float(v), 6) for v in bx_curve],
        evidence_curve=[round(float(v), 6) for v in evidence_curve],
    )


def summarize_abx(results: List[AbxTrialResult]) -> Dict[str, float | int]:
    """Aggregate one model/layer's ABX outcomes."""

    if not results:
        return {
            "n_trials": 0,
            "abx_error_rate": float("nan"),
            "abx_accuracy": float("nan"),
            "mean_margin": float("nan"),
            "std_margin": float("nan"),
        }
    margins = np.asarray([item.margin for item in results], dtype=float)
    accuracy = float(np.mean([item.correct for item in results]))
    return {
        "n_trials": int(len(results)),
        "abx_error_rate": round(1.0 - accuracy, 6),
        "abx_accuracy": round(accuracy, 6),
        "mean_margin": round(float(np.mean(margins)), 6),
        "std_margin": round(float(np.std(margins)), 6),
    }


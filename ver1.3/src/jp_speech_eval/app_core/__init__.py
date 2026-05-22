"""Small C-end practice layer built on top of the v1.3 evaluator.

The modules in this package intentionally do not change the acoustic scoring
core.  They turn existing evaluator evidence into user profile, progress, and
practice-flow objects that are easier to use in a consumer app.
"""

from .calibration import build_voice_profile
from .personalized_scorer import compare_to_profile
from .practice_modes import compute_reference_dependency_gap, step_label
from .progress_tracker import append_progress_record, load_progress_records
from .user_profile import UserVoiceProfile, load_user_profile, save_user_profile

__all__ = [
    "UserVoiceProfile",
    "append_progress_record",
    "build_voice_profile",
    "compare_to_profile",
    "compute_reference_dependency_gap",
    "load_progress_records",
    "load_user_profile",
    "save_user_profile",
    "step_label",
]

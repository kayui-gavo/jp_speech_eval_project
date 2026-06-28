from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLED_ASSET_ROOT = PACKAGE_ROOT / "resources"
EDITABLE_PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def resolve_asset_path(relative_path: str | Path) -> Path:
    """Resolve a runtime asset in both wheel and editable installations.

    Runtime JSON files are bundled below ``jp_speech_eval/resources``.  The
    editable-project fallback preserves compatibility with existing scripts
    that keep the same files under ``ver1.3/configs`` or ``ver1.3/results``.
    """

    relative = Path(relative_path)
    if relative.is_absolute():
        return relative

    bundled = BUNDLED_ASSET_ROOT / relative
    if bundled.exists():
        return bundled

    editable = EDITABLE_PROJECT_ROOT / relative
    if editable.exists():
        return editable

    # Return the portable location so a missing-file error is deterministic.
    return bundled

"""Small, deterministic helpers for content-cascade experiment splits."""

from __future__ import annotations

from typing import Iterable, Mapping


def split_rows_by_target_sentence(
    rows: Iterable[Mapping[str, str]], development_target_ids: set[str]
) -> tuple[list[Mapping[str, str]], list[Mapping[str, str]]]:
    """Return target-disjoint development and held-out rows.

    The function intentionally knows nothing about speaker identity: callers
    must report speaker overlap separately instead of mislabelling this as a
    speaker-disjoint evaluation.
    """
    development: list[Mapping[str, str]] = []
    held_out: list[Mapping[str, str]] = []
    for row in rows:
        target_id = str(row["target_sentence_id"])
        (development if target_id in development_target_ids else held_out).append(row)
    if {str(row["target_sentence_id"]) for row in development} & {
        str(row["target_sentence_id"]) for row in held_out
    }:
        raise AssertionError("target sentence split is not disjoint")
    return development, held_out

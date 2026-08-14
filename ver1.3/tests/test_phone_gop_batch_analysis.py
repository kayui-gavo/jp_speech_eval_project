from __future__ import annotations

from jp_speech_eval.phone_gop_batch_analysis import analyze_manual_gop_batch, resolve_focus_indices


def _payload(b_margin: float, a_margin: float = 3.0, c_margin: float = 3.5, competitor: str = "c"):
    def row(index: int, phone: str, margin: float, comp: str):
        return {
            "phone_index": index,
            "canonical_phone": phone,
            "mean_logit_margin": margin + 0.4,
            "max_logit_margin": margin + 0.7,
            "posterior_gop_margin": margin,
            "mean_entropy": 0.2,
            "best_competitor_phone": comp,
        }

    return {
        "schema": "japanese_phone_gop_shadow_v1",
        "gop": {
            "evidence": [
                row(0, "a", a_margin, "b"),
                row(1, "b", b_margin, competitor),
                row(2, "c", c_margin, "b"),
            ]
        },
    }


def test_focus_resolution_supports_second_phone_and_whole() -> None:
    rows = [
        {"canonical_phone": "a"},
        {"canonical_phone": "b"},
        {"canonical_phone": "a"},
    ]
    assert resolve_focus_indices(rows, "second_a") == [2]
    assert resolve_focus_indices(rows, "b") == [1]
    assert resolve_focus_indices(rows, "whole") == [0, 1, 2]


def test_batch_analysis_uses_clean_repeat_variance_before_error_delta() -> None:
    manifest = [
        {
            "clip_id": "X_N1",
            "repeat_group": "X",
            "component": "segmental",
            "target_text": "x",
            "target_focus": "b",
            "condition": "normal",
            "target_error_type": "none",
            "expected_competitor": "",
        },
        {
            "clip_id": "X_N2",
            "repeat_group": "X",
            "component": "segmental",
            "target_text": "x",
            "target_focus": "b",
            "condition": "normal_repeat",
            "target_error_type": "none",
            "expected_competitor": "",
        },
        {
            "clip_id": "X_E",
            "repeat_group": "X",
            "component": "segmental",
            "target_text": "x",
            "target_focus": "b",
            "condition": "error",
            "target_error_type": "substitution",
            "expected_competitor": "c",
        },
    ]
    payloads = {
        "X_N1": _payload(4.0),
        "X_N2": _payload(3.8),
        "X_E": _payload(-2.0, a_margin=2.9, c_margin=3.4, competitor="c"),
    }
    report = analyze_manual_gop_batch(manifest, payloads)
    assert report["complete_error_groups"] == 1
    assert report["directional_error_groups_evaluated"] == 1
    assert report["directional_error_groups_with_lower_posterior_margin"] == 1
    group = report["groups"][0]
    assert abs(group["clean_repeat_absolute_delta"]["posterior_gop_margin"] - 0.2) < 1e-8
    error = group["errors"][0]
    assert error["error_minus_clean_baseline"]["posterior_gop_margin"] < -5.0
    assert error["posterior_error_to_clean_repeat_ratio"] > 20
    assert error["expected_competitor_match"] is True
    # Neighbor /a,c/ margins barely changed, so localization leakage remains small.
    assert error["neighbor_posterior_delta_max"] < 0.2


def test_missing_files_are_reported_without_fabricating_results() -> None:
    manifest = [
        {
            "clip_id": "MISSING",
            "repeat_group": "M",
            "component": "segmental",
            "target_text": "x",
            "target_focus": "b",
            "condition": "error",
        }
    ]
    report = analyze_manual_gop_batch(manifest, {})
    assert report["groups_analyzed"] == 0
    assert report["missing_clip_count"] == 1
    assert report["missing_clips"] == ["MISSING"]
    assert report["score_mapped"] is False

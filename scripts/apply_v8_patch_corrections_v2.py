from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one match in {path}, got {count}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Existing language-safety test predates word-timestamp support. Preserve its
# actual assertion (language=None) while accepting the newer optional kwarg.
replace_once(
    "ver1.3/tests/test_language_safe_asr_confirmation.py",
    '        def fake_try(y, sr, model_name, language="ja"):\n',
    '        def fake_try(y, sr, model_name, language="ja", *, word_timestamps=False):\n',
)

# v7 made direct free speech a core/public mode. Bring the stale UI contract
# test in line with the established Space behavior instead of removing the mode.
replace_once(
    "ver1.3/tests/test_package_and_ui_contract.py",
    '''        self.assertEqual(debug_ui.CORE_MODES, ["reference", "asr_pseudo_reference", "kanade_asr_voice_reference"])
        self.assertEqual(debug_ui.PUBLIC_DEMO_MODES, ["reference", "asr_pseudo_reference", "kanade_asr_voice_reference"])
        self.assertNotIn("transcript_assisted_light", debug_ui.CORE_MODES)
        self.assertNotIn("acoustic", debug_ui.PUBLIC_DEMO_MODES)
''',
    '''        self.assertEqual(
            debug_ui.CORE_MODES,
            ["reference", "transcript_assisted_light", "asr_pseudo_reference", "kanade_asr_voice_reference"],
        )
        self.assertEqual(debug_ui.PUBLIC_DEMO_MODES, ["reference", "transcript_assisted_light"])
        self.assertIn("transcript_assisted_light", debug_ui.CORE_MODES)
        self.assertNotIn("acoustic", debug_ui.PUBLIC_DEMO_MODES)
        self.assertNotIn("asr_pseudo_reference", debug_ui.PUBLIC_DEMO_MODES)
        self.assertNotIn("kanade_asr_voice_reference", debug_ui.PUBLIC_DEMO_MODES)
''',
)

# Special-mora limited candidate is now shadow-only by default. Tests that
# exercise the learner-facing candidate must opt in explicitly.
replace_once(
    "ver1.3/tests/test_product_guardrails.py",
    '''        ]))
        self.assertFalse(rendered["display_total_score"])
        self.assertEqual(rendered["focus_feedback"]["category"], "special_mora")
''',
    '''        ]), enable_user_facing_calibrated_special_mora=True)
        self.assertFalse(rendered["display_total_score"])
        self.assertEqual(rendered["focus_feedback"]["category"], "special_mora")
''',
)
replace_once(
    "ver1.3/tests/test_product_guardrails.py",
    '''    def test_v2_limited_candidate_emits_allowed_types_by_default(self) -> None:
''',
    '''    def test_v2_limited_candidate_emits_allowed_types_when_explicitly_enabled(self) -> None:
''',
)
replace_once(
    "ver1.3/tests/test_product_guardrails.py",
    '''        rendered = render_user_facing_result(result, special_mora_threshold_profile="v2_limited_candidate")
        self.assertTrue(any(item["user_feedback_allowed"] for item in rendered["debug"]["special_mora_decisions"]))
''',
    '''        rendered = render_user_facing_result(
            result,
            special_mora_threshold_profile="v2_limited_candidate",
            enable_user_facing_calibrated_special_mora=True,
        )
        self.assertTrue(any(item["user_feedback_allowed"] for item in rendered["debug"]["special_mora_decisions"]))
''',
)

# Demo smoke semantics follow the same safe default: the candidate stays in
# debug/shadow unless explicitly enabled, while the explicit scenario still
# verifies the learner-facing path can be exercised intentionally.
replace_once(
    "ver1.3/scripts/run_demo_flow_smoke_tests.py",
    '''        {"name": "fixed_clear_long_vowel_default_gentle", "result": clear_short, "kwargs": {"special_mora_threshold_profile": "v2_limited_candidate"}, "expect": {"status": "practice_suggestion", "suggestion_type": "special_mora"}},
''',
    '''        {"name": "fixed_clear_long_vowel_default_shadow_only", "result": clear_short, "kwargs": {"special_mora_threshold_profile": "v2_limited_candidate"}, "expect": {"no_special_feedback": True}},
''',
)

# An optional rhythm derivative must not invalidate an otherwise valid SSL
# pronunciation/reference-distance shadow. Short feature fixtures and short real
# utterances can have a valid cosine-DTW distance but too few frames for the
# five-frame rhythm smoothing window.
replace_once(
    "ver1.3/src/jp_speech_eval/shadow_assessment.py",
    '''                rhythm = tempo_irregularity_from_dtw_path(
                    path,
                    reference_frame_count=int(alignment["reference_frame_count"]),
                    smoothing_frames=5,
                )
                reference_rhythm.append(
                    {
                        "reference_id": reference_row["reference_id"],
                        "tempo_irregularity_rad": float(rhythm["tempo_irregularity_rad"]),
                        "global_frame_duration_ratio": (
                            float(alignment["user_frame_count"])
                            / max(float(alignment["reference_frame_count"]), 1.0)
                        ),
                        "angle_count": int(rhythm["angle_count"]),
                    }
                )
''',
    '''                try:
                    rhythm = tempo_irregularity_from_dtw_path(
                        path,
                        reference_frame_count=int(alignment["reference_frame_count"]),
                        smoothing_frames=5,
                    )
                except (ValueError, FloatingPointError):
                    rhythm = None
                if rhythm is not None:
                    reference_rhythm.append(
                        {
                            "reference_id": reference_row["reference_id"],
                            "tempo_irregularity_rad": float(rhythm["tempo_irregularity_rad"]),
                            "global_frame_duration_ratio": (
                                float(alignment["user_frame_count"])
                                / max(float(alignment["reference_frame_count"]), 1.0)
                            ),
                            "angle_count": int(rhythm["angle_count"]),
                        }
                    )
''',
)
replace_once(
    "ver1.3/src/jp_speech_eval/shadow_assessment.py",
    '''            aggregate_irregularity = aggregate_reference_distances(
                [item["tempo_irregularity_rad"] for item in reference_rhythm],
                strategy=ssl_reference_aggregation,
            )
            aggregate_duration_ratio = aggregate_reference_distances(
                [item["global_frame_duration_ratio"] for item in reference_rhythm],
                strategy=ssl_reference_aggregation,
            )
''',
    '''            aggregate_irregularity = (
                aggregate_reference_distances(
                    [item["tempo_irregularity_rad"] for item in reference_rhythm],
                    strategy=ssl_reference_aggregation,
                )
                if reference_rhythm
                else None
            )
            aggregate_duration_ratio = (
                aggregate_reference_distances(
                    [item["global_frame_duration_ratio"] for item in reference_rhythm],
                    strategy=ssl_reference_aggregation,
                )
                if reference_rhythm
                else None
            )
''',
)
old_rhythm_payload = '''            shadow["rhythm_dtw_v1"] = {
                "available": True,
                "backend": "wavlm_cosine_dtw_warp_path",
                "model_id": ssl_model_id,
                "layer": ssl_layer,
                "reference_id": reference_identity,
                "reference_panel": panel_meta,
                "reference_count": len(reference_rhythm),
                "aggregate_strategy": ssl_reference_aggregation,
                "tempo_irregularity_rad": float(aggregate_irregularity),
                "tempo_irregularity_deg": float(np.degrees(aggregate_irregularity)),
                "global_frame_duration_ratio": float(aggregate_duration_ratio),
                "reference_rhythm": reference_rhythm,
                "interpretation": "lower_is_more_locally_uniform_relative_tempo_shadow_only",
                "score_mapped": False,
                "product_calibrated": False,
                "user_facing": False,
                "latency_ms": round(elapsed * 1000.0, 3),
            }
'''
new_rhythm_payload = '''            if reference_rhythm:
                shadow["rhythm_dtw_v1"] = {
                    "available": True,
                    "backend": "wavlm_cosine_dtw_warp_path",
                    "model_id": ssl_model_id,
                    "layer": ssl_layer,
                    "reference_id": reference_identity,
                    "reference_panel": panel_meta,
                    "reference_count": len(reference_rhythm),
                    "aggregate_strategy": ssl_reference_aggregation,
                    "tempo_irregularity_rad": float(aggregate_irregularity),
                    "tempo_irregularity_deg": float(np.degrees(aggregate_irregularity)),
                    "global_frame_duration_ratio": float(aggregate_duration_ratio),
                    "reference_rhythm": reference_rhythm,
                    "interpretation": "lower_is_more_locally_uniform_relative_tempo_shadow_only",
                    "score_mapped": False,
                    "product_calibrated": False,
                    "user_facing": False,
                    "latency_ms": round(elapsed * 1000.0, 3),
                }
            else:
                shadow["rhythm_dtw_v1"] = {
                    "available": False,
                    "backend": "wavlm_cosine_dtw_warp_path",
                    "model_id": ssl_model_id,
                    "layer": ssl_layer,
                    "reference_id": reference_identity,
                    "reference_panel": panel_meta,
                    "reference_count": 0,
                    "reason": "insufficient_dtw_frames_for_rhythm_metric",
                    "score_mapped": False,
                    "product_calibrated": False,
                    "user_facing": False,
                    "latency_ms": round(elapsed * 1000.0, 3),
                }
'''
replace_once(
    "ver1.3/src/jp_speech_eval/shadow_assessment.py",
    old_rhythm_payload,
    new_rhythm_payload,
)

# Lock the intended failure isolation explicitly.
replace_once(
    "ver1.3/tests/test_shadow_assessment.py",
    '''    assert result["details"]["shadow"]["ssl_pronunciation"]["available"] is True


def test_unified_result_preserves_zero_values():
''',
    '''    assert result["details"]["shadow"]["ssl_pronunciation"]["available"] is True
    assert result["details"]["shadow"]["rhythm_dtw_v1"]["available"] is False
    assert result["details"]["shadow"]["rhythm_dtw_v1"]["reason"] == "insufficient_dtw_frames_for_rhythm_metric"


def test_unified_result_preserves_zero_values():
''',
)

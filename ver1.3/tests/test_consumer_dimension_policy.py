from __future__ import annotations

import unittest

from jp_speech_eval.consumer_dimension_policy import (
    build_consumer_score_components,
    build_consumer_score_dimensions,
)


class ConsumerDimensionPolicyTest(unittest.TestCase):
    def _result(self) -> dict:
        return {
            "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
            "pronunciation_score": 78,
            "fluency_score": 84,
            "prosody_score": 81,
            "mora_table": [
                {"mora": "ラ", "f0_hz": 180.0},
                {"mora": "ー", "f0_hz": 193.0},
                {"mora": "メ", "f0_hz": 205.0},
                {"mora": "ン", "f0_hz": 198.0},
                {"mora": "ヲ", "f0_hz": 188.0},
                {"mora": "ク", "f0_hz": 177.0},
                {"mora": "ダ", "f0_hz": 168.0},
                {"mora": "サ", "f0_hz": 162.0},
                {"mora": "イ", "f0_hz": 158.0},
            ],
            "details": {
                "pronunciation": {
                    "score_interpretation": "mora_timing_proxy_not_full_segmental_pronunciation",
                },
                "fluency": {
                    "delivery_fluency_score": 88,
                    "rhythm_timing_score": 82,
                    "rate_score": 80,
                    "pause_score": 90,
                },
                "prosody": {
                    "contour_valid_mora_count": 8,
                    "mora_count": 9,
                    "contour_corr": 0.65,
                    "note": "ok",
                },
                "reliability": {
                    "f0_coverage": 0.88,
                    "endpointing": 1.0,
                    "duration_ratio_to_reference": 1.05,
                },
                "alignment": {
                    "available": True,
                    "used_equal_fallback": False,
                    "normalized_dtw_cost": 3.8,
                },
                "content_match": {
                    "status": "pass",
                    "score": 0.90,
                    "kana_similarity": 0.95,
                    "duration_ratio": 1.05,
                    "content_verified": True,
                },
                "reference_f0_by_mora": [178.0, 190.0, 202.0, 196.0, 186.0, 175.0, 166.0, 160.0, 156.0],
                "tone": {
                    "pitch_range_log": 0.30,
                    "pitch_score": 90,
                },
                "shadow": {},
            },
        }

    def _dims(self, result: dict | None = None, display_score: int | None = 80) -> list[dict]:
        return build_consumer_score_dimensions(
            result or self._result(), {"display_score": display_score}, mode="reference"
        )

    def test_four_dimensions_are_stable_numeric_and_non_overlapping(self) -> None:
        dims = self._dims()
        self.assertEqual(
            [item["key"] for item in dims],
            ["delivery_fluency", "clarity", "mora_timing", "intonation"],
        )
        self.assertEqual(
            [item["label"] for item in dims],
            ["流暢さ", "明瞭さ", "リズム", "抑揚"],
        )
        self.assertTrue(all(item["available"] for item in dims))
        self.assertTrue(all(isinstance(item["value"], int) for item in dims))
        self.assertTrue(all(0 <= item["value"] <= 100 for item in dims))
        self.assertTrue(all(item["product_calibrated"] is False for item in dims))
        self.assertNotIn("韻律", [item["label"] for item in dims])
        self.assertNotIn("pitch_accent", {item["key"] for item in dims})

    def test_fluency_uses_speed_and_pause_not_pause_only(self) -> None:
        by_key = {item["key"]: item for item in self._dims()}
        self.assertEqual(by_key["delivery_fluency"]["value"], 85)
        self.assertEqual(by_key["delivery_fluency"]["evidence_tier"], "rate_plus_pause")

    def test_clarity_prefers_asr_target_agreement_without_reusing_timing_proxy(self) -> None:
        by_key = {item["key"]: item for item in self._dims()}
        clarity = by_key["clarity"]
        self.assertTrue(clarity["available"])
        self.assertGreater(clarity["value"], 80)
        self.assertEqual(clarity["evidence_tier"], "asr_target_agreement")
        self.assertNotEqual(clarity["source_field"], "pronunciation_score")
        self.assertIn("machine-intelligibility", clarity["note"])

    def test_mapped_ssl_overrides_lower_tier_clarity_proxy(self) -> None:
        result = self._result()
        result["details"]["shadow"]["ssl_pronunciation"] = {
            "score_mapped": True,
            "mapped_score": 73,
        }
        by_key = {item["key"]: item for item in self._dims(result)}
        self.assertEqual(by_key["clarity"]["value"], 73)
        self.assertEqual(by_key["clarity"]["evidence_tier"], "mapped_pronunciation_evidence")
        self.assertIn("ssl", by_key["clarity"]["construct"])

    def test_rhythm_survives_equal_alignment_fallback(self) -> None:
        result = self._result()
        result["details"]["alignment"].update({
            "available": False,
            "used_equal_fallback": True,
            "normalized_dtw_cost": None,
        })
        by_key = {item["key"]: item for item in self._dims(result)}
        rhythm = by_key["mora_timing"]
        self.assertTrue(rhythm["available"])
        self.assertIsInstance(rhythm["value"], int)
        self.assertEqual(rhythm["confidence"], "low")
        self.assertEqual(rhythm["evidence_tier"], "alignment_fallback_broad_timing")
        self.assertIn("local alignment fell back", rhythm["note"])

    def test_top_level_alignment_mode_alone_disables_local_timing_and_f0(self) -> None:
        result = self._result()
        # Simulates an older evaluator where only the top-level field records
        # fallback and the nested alignment object still looks nominal.
        result["alignment_mode"] = "cached_dtw_fallback_equal"
        result["details"]["alignment"].update({
            "available": True,
            "used_equal_fallback": False,
            "normalized_dtw_cost": 3.5,
        })
        by_key = {item["key"]: item for item in self._dims(result)}
        self.assertEqual(by_key["mora_timing"]["confidence"], "low")
        self.assertEqual(by_key["mora_timing"]["evidence_tier"], "alignment_fallback_broad_timing")
        self.assertNotEqual(by_key["mora_timing"]["source_field"], "pronunciation_score+duration_ratio_to_reference")
        self.assertEqual(by_key["intonation"]["confidence"], "low")
        self.assertEqual(by_key["intonation"]["evidence_tier"], "alignment_fallback_pitch_range_proxy")
        self.assertNotEqual(by_key["intonation"]["value"], result["prosody_score"])

    def test_primary_reference_f0_contour_gets_intonation_score(self) -> None:
        by_key = {item["key"]: item for item in self._dims()}
        intonation = by_key["intonation"]
        self.assertEqual(intonation["value"], 81)
        self.assertEqual(intonation["evidence_tier"], "mora_contour_primary")
        self.assertIn("not strict lexical pitch-accent", intonation["note"])

    def test_intonation_survives_prosody_gate_failure_with_partial_f0(self) -> None:
        result = self._result()
        result["prosody_score"] = 50
        result["details"]["reliability"]["f0_coverage"] = 0.2
        result["details"]["prosody"]["contour_valid_mora_count"] = 2
        result["details"]["prosody"]["contour_corr"] = None
        result["details"]["prosody"]["note"] = "insufficient_valid_mora_f0"
        for row in result["mora_table"][2:]:
            row["f0_hz"] = None
        by_key = {item["key"]: item for item in self._dims(result)}
        intonation = by_key["intonation"]
        self.assertTrue(intonation["available"])
        self.assertIsInstance(intonation["value"], int)
        self.assertEqual(intonation["evidence_tier"], "partial_f0_fallback")
        self.assertEqual(intonation["confidence"], "low")

    def test_intonation_has_low_confidence_prior_when_f0_is_completely_missing(self) -> None:
        result = self._result()
        result["prosody_score"] = 50
        result["details"]["prosody"].update({
            "contour_valid_mora_count": 0,
            "contour_corr": None,
            "note": "no_valid_f0",
        })
        result["details"]["tone"] = {"pitch_range_log": None, "pitch_score": 80}
        for row in result["mora_table"]:
            row["f0_hz"] = None
        by_key = {item["key"]: item for item in self._dims(result)}
        intonation = by_key["intonation"]
        self.assertTrue(intonation["available"])
        self.assertEqual(intonation["value"], 70)
        self.assertEqual(intonation["evidence_tier"], "prior_fallback")
        self.assertEqual(intonation["confidence"], "low")

    def test_valid_off_target_japanese_disables_target_relative_dimension_evidence(self) -> None:
        result = self._result()
        result["details"]["content_match"].update({
            "status": "fail",
            "kana_similarity": 0.05,
            "score": 0.10,
            "duration_ratio": 2.0,
        })
        # Deliberately make target-relative signals awful; they must not punish a
        # user who spoke a different but valid Japanese sentence.
        result["pronunciation_score"] = 15
        result["prosody_score"] = 12
        result["details"]["reliability"]["duration_ratio_to_reference"] = 2.0
        result["details"]["fluency"]["rate_score"] = 82
        result["details"]["tone"] = {"pitch_range_log": 0.25, "pitch_score": 84}

        components = build_consumer_score_components(result, mode="reference")
        by_key = {item["key"]: item for item in components}
        self.assertEqual(by_key["clarity"]["value"], 70)
        self.assertEqual(by_key["clarity"]["evidence_tier"], "reference_independent_prior_fallback")
        self.assertNotIn("kana_similarity", by_key["clarity"]["source_field"])
        self.assertEqual(by_key["mora_timing"]["value"], 82)
        self.assertEqual(by_key["mora_timing"]["evidence_tier"], "reference_independent_rate_fallback")
        self.assertNotIn("duration_ratio", by_key["mora_timing"]["source_field"])
        self.assertNotEqual(by_key["intonation"]["value"], 12)
        self.assertEqual(by_key["intonation"]["evidence_tier"], "pitch_range_fallback")
        self.assertIn("target-relative evidence disabled", by_key["intonation"]["note"])

    def test_true_no_score_input_keeps_all_dimensions_unavailable(self) -> None:
        dims = self._dims(display_score=None)
        self.assertEqual(len(dims), 4)
        self.assertTrue(all(not item["available"] for item in dims))
        self.assertTrue(all(item["value"] is None for item in dims))


if __name__ == "__main__":
    unittest.main()

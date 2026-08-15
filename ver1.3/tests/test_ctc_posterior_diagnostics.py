from __future__ import annotations

import unittest

import numpy as np

from jp_speech_eval.ctc_posterior_diagnostics import compute_ctc_posterior_diagnostics


class CtcPosteriorDiagnosticsTest(unittest.TestCase):
    def test_peaky_blank_frames_are_visible_without_becoming_score(self) -> None:
        logits = np.asarray(
            [
                [8.0, 0.0, 0.0],
                [0.0, 8.0, 0.0],
                [8.0, 0.0, 0.0],
                [0.0, 0.0, 8.0],
            ],
            dtype=np.float64,
        )
        result = compute_ctc_posterior_diagnostics(
            logits,
            blank_id=0,
            phone_token_ids=[1, 2],
        )
        self.assertTrue(result.available)
        self.assertEqual(result.schema, "ctc_posterior_diagnostics_v2")
        self.assertFalse(result.product_calibrated)
        self.assertFalse(result.score_mapped)
        self.assertAlmostEqual(result.blank_top1_fraction, 0.5)
        self.assertGreater(result.top1_posterior_mean, 0.99)
        self.assertEqual(result.phone_token_count, 2)
        self.assertGreater(result.full_top1_logit_margin_mean, 7.9)
        self.assertGreater(result.phone_top1_logit_margin_mean, 7.9)
        self.assertTrue(result.summary["logit_margin_diagnostics_available"])
        self.assertFalse(result.summary["logit_margin_is_pronunciation_error"])
        self.assertFalse(result.summary["universal_threshold_defined"])
        self.assertFalse(result.summary["heuristic_alert_thresholds_defined"])
        self.assertFalse(result.summary["individual_frame_entropy_is_pronunciation_error"])
        self.assertEqual(result.warnings, [])

    def test_diffuse_logits_have_higher_entropy_and_lower_logit_margin_than_peaky_logits(self) -> None:
        diffuse = np.zeros((5, 4), dtype=np.float64)
        peaky = np.asarray([[10.0, 0.0, 0.0, 0.0]] * 5, dtype=np.float64)
        diffuse_result = compute_ctc_posterior_diagnostics(
            diffuse,
            blank_id=0,
            phone_token_ids=[1, 2, 3],
        )
        peaky_result = compute_ctc_posterior_diagnostics(
            peaky,
            blank_id=0,
            phone_token_ids=[1, 2, 3],
        )
        self.assertGreater(
            diffuse_result.full_normalized_entropy_mean,
            peaky_result.full_normalized_entropy_mean,
        )
        self.assertLess(diffuse_result.top1_posterior_mean, peaky_result.top1_posterior_mean)
        self.assertLess(diffuse_result.full_top1_logit_margin_mean, peaky_result.full_top1_logit_margin_mean)

    def test_control_tokens_can_be_excluded_by_phone_inventory_without_alert_threshold(self) -> None:
        logits = np.asarray([[0.0, 2.0, 1.0, 7.0]], dtype=np.float64)
        result = compute_ctc_posterior_diagnostics(
            logits,
            blank_id=0,
            phone_token_ids=[1, 2],
        )
        self.assertEqual(result.phone_token_count, 2)
        # Token 3 dominates the frame but is deliberately outside phone mass.
        self.assertLess(result.phone_mass_mean, 0.05)
        self.assertAlmostEqual(result.phone_top1_logit_margin_mean, 1.0)
        self.assertEqual(result.warnings, [])
        self.assertFalse(result.summary["heuristic_alert_thresholds_defined"])

    def test_logit_margins_are_invariant_to_additive_frame_offsets(self) -> None:
        logits = np.asarray(
            [
                [0.0, 2.0, 1.0],
                [1.0, 0.5, 3.0],
            ],
            dtype=np.float64,
        )
        shifted = logits + np.asarray([[100.0], [-30.0]])
        a = compute_ctc_posterior_diagnostics(logits, blank_id=0, phone_token_ids=[1, 2])
        b = compute_ctc_posterior_diagnostics(shifted, blank_id=0, phone_token_ids=[1, 2])
        self.assertAlmostEqual(a.full_top1_logit_margin_mean, b.full_top1_logit_margin_mean)
        self.assertAlmostEqual(a.phone_top1_logit_margin_mean, b.phone_top1_logit_margin_mean)
        self.assertAlmostEqual(
            a.blank_minus_phone_top_logit_margin_mean,
            b.blank_minus_phone_top_logit_margin_mean,
        )

    def test_invalid_phone_inventory_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "phone token inventory is empty"):
            compute_ctc_posterior_diagnostics(
                np.zeros((2, 3), dtype=np.float64),
                blank_id=0,
                phone_token_ids=[0],
            )


if __name__ == "__main__":
    unittest.main()
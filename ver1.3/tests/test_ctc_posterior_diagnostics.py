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
        self.assertFalse(result.product_calibrated)
        self.assertFalse(result.score_mapped)
        self.assertAlmostEqual(result.blank_top1_fraction, 0.5)
        self.assertGreater(result.top1_posterior_mean, 0.99)
        self.assertEqual(result.phone_token_count, 2)
        self.assertFalse(result.summary["universal_threshold_defined"])
        self.assertFalse(result.summary["individual_frame_entropy_is_pronunciation_error"])

    def test_diffuse_logits_have_higher_entropy_than_peaky_logits(self) -> None:
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

    def test_control_tokens_can_be_excluded_by_phone_inventory(self) -> None:
        logits = np.asarray([[0.0, 2.0, 1.0, 7.0]], dtype=np.float64)
        result = compute_ctc_posterior_diagnostics(
            logits,
            blank_id=0,
            phone_token_ids=[1, 2],
        )
        self.assertEqual(result.phone_token_count, 2)
        # Token 3 dominates the frame but is deliberately outside phone mass.
        self.assertLess(result.phone_mass_mean, 0.05)
        self.assertIn("low_mean_phone_probability_mass", result.warnings)

    def test_invalid_phone_inventory_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "phone token inventory is empty"):
            compute_ctc_posterior_diagnostics(
                np.zeros((2, 3), dtype=np.float64),
                blank_id=0,
                phone_token_ids=[0],
            )


if __name__ == "__main__":
    unittest.main()

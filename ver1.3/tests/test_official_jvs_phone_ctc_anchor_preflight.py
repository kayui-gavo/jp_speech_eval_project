import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_official_jvs_phone_ctc_anchor_preflight.py"
SPEC = importlib.util.spec_from_file_location("jvs_anchor_preflight", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OfficialJvsPhoneCtcAnchorPreflightTest(unittest.TestCase):
    def test_rotation_preserves_phone_count_and_changes_order(self) -> None:
        source = [1, 2, 3, 4, 5, 6]
        rotated = MODULE._rotate_control(source)
        self.assertEqual(len(rotated), len(source))
        self.assertCountEqual(rotated, source)
        self.assertNotEqual(rotated, source)

    def test_sequence_metrics_favor_matching_ctc_path(self) -> None:
        # vocab: blank=0, a=1, b=2.  A five-frame CTC path strongly supports
        # a,b while the same-length rotated control b,a is much less likely.
        logits = np.asarray(
            [
                [7.0, 0.0, 0.0],
                [0.0, 7.0, 0.0],
                [7.0, 0.0, 0.0],
                [0.0, 0.0, 7.0],
                [7.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        result = MODULE._sequence_metrics(
            logits,
            ["a", "b"],
            {"<blank>": 0, "a": 1, "b": 2},
            0,
        )
        self.assertTrue(result["available"])
        self.assertGreater(result["canonical_minus_permuted"], 0.0)
        self.assertGreater(result["canonical_minus_permuted_per_frame"], 0.0)
        self.assertEqual(result["phone_count"], 2)
        self.assertFalse(result["control_is_pronunciation_error_label"])

    def test_sequence_metrics_reports_phone_inventory_mismatch(self) -> None:
        result = MODULE._sequence_metrics(
            np.zeros((5, 3), dtype=np.float64),
            ["a", "missing"],
            {"<blank>": 0, "a": 1, "b": 2},
            0,
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "phone_inventory_mismatch")
        self.assertEqual(result["missing_phones"], ["missing"])


if __name__ == "__main__":
    unittest.main()

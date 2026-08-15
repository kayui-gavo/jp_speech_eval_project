from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_multibackbone_controlled_phone_edits",
    ROOT / "scripts" / "run_multibackbone_controlled_phone_edits.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MultiBackboneControlDefinitionTest(unittest.TestCase):
    def test_every_control_is_exactly_one_intended_phone_edit(self) -> None:
        expected = {
            "voicing_b_p": ("substitution", "b", "p"),
            "voicing_g_k": ("substitution", "g", "k"),
            "affricate_ts_s": ("substitution", "ts", "s"),
            "fricative_sh_s": ("substitution", "sh", "s"),
            "affricate_ch_sh": ("substitution", "ch", "sh"),
            "sokuon_deletion": ("deletion", "cl", None),
            "mora_n_deletion": ("deletion", "N", None),
        }
        self.assertEqual({row["case_id"] for row in MODULE.CASES}, set(expected))
        for case in MODULE.CASES:
            target = MODULE._phones(case["target"])
            error = MODULE._phones(case["spoken_error"])
            edit = MODULE._single_edit(target, error, case["edit_type"])
            self.assertTrue(edit["available"], msg=f"{case['case_id']}: {edit}")
            edit_type, target_phone, error_phone = expected[case["case_id"]]
            self.assertEqual(edit["edit_type"], edit_type)
            self.assertEqual(edit["target_phone"], target_phone)
            self.assertEqual(edit["error_phone"], error_phone)

    def test_no_control_claims_a_learner_label(self) -> None:
        # Script-level contract: the experiment is synthetic directionality,
        # not Japanese-L2 pronunciation criterion validity.
        source = (ROOT / "scripts" / "run_multibackbone_controlled_phone_edits.py").read_text(encoding="utf-8")
        self.assertIn('"synthetic_control_is_learner_validity": False', source)
        self.assertIn('"global_lpr_zero_threshold_validated": False', source)
        self.assertIn('"product_score_changed": False', source)


if __name__ == "__main__":
    unittest.main()

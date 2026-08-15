from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_controlled_tts_phone_edit_preflight.py"
SPEC = importlib.util.spec_from_file_location("controlled_tts_contract", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _resolved_edit(case: dict, target_phones: list[str], target_roles: dict, spoken_phones: list[str]) -> dict:
    raw = MODULE._single_edit(target_phones, spoken_phones, case["edit_type"])
    return MODULE._resolve_edit_for_construct(
        raw,
        target_phones,
        target_roles,
        case["expected_construct_role"],
    )


class ControlledTtsConstructContractTest(unittest.TestCase):
    def test_every_control_edit_hits_declared_construct(self) -> None:
        seen = set()
        for case in MODULE.CASES:
            with self.subTest(case_id=case["case_id"]):
                target_phones, target_roles = MODULE._target_contract(case["target"])
                spoken_phones = MODULE._phones(case["spoken_error"])
                edit = _resolved_edit(case, target_phones, target_roles, spoken_phones)
                self.assertTrue(target_roles["available"], target_roles.get("summary"))
                self.assertTrue(edit["available"], edit)
                index = int(edit["target_index"])
                role = target_roles["roles"][index]
                self.assertEqual(role, case["expected_construct_role"])
                self.assertEqual(edit["target_construct_role"], role)
                seen.add(role)
        self.assertEqual(
            seen,
            {MODULE.ORDINARY_ROLE, MODULE.SPECIAL_MORA_ROLE, MODULE.LONG_VOWEL_ROLE},
        )

    def test_long_vowel_control_resolves_sequence_equivalent_deletion_by_role(self) -> None:
        case = next(row for row in MODULE.CASES if row["case_id"] == "long_vowel_deletion")
        target_phones, target_roles = MODULE._target_contract(case["target"])
        spoken_phones = MODULE._phones(case["spoken_error"])
        raw = MODULE._single_edit(target_phones, spoken_phones, "deletion")
        self.assertFalse(raw["available"])
        self.assertTrue(raw["sequence_edit_ambiguous"])
        self.assertEqual(len(raw["candidate_indices"]), 2)

        edit = MODULE._resolve_edit_for_construct(
            raw,
            target_phones,
            target_roles,
            MODULE.LONG_VOWEL_ROLE,
        )
        self.assertTrue(edit["available"], edit)
        self.assertTrue(edit["sequence_edit_ambiguous"])
        self.assertTrue(edit["resolved_by_construct_role"])
        self.assertEqual(edit["resolution_basis"], "target_construct_role_not_acoustic_phone_identity")
        index = int(edit["target_index"])
        self.assertEqual(target_roles["roles"][index], MODULE.LONG_VOWEL_ROLE)
        self.assertIn(str(edit["target_phone"]), {"a", "i", "u", "e", "o"})

    def test_ambiguous_deletion_without_unique_role_match_stays_unavailable(self) -> None:
        target = ["a", "a"]
        spoken = ["a"]
        raw = MODULE._single_edit(target, spoken, "deletion")
        roles = {
            "available": True,
            "roles": [MODULE.ORDINARY_ROLE, MODULE.ORDINARY_ROLE],
        }
        edit = MODULE._resolve_edit_for_construct(
            raw,
            target,
            roles,
            MODULE.ORDINARY_ROLE,
        )
        self.assertFalse(edit["available"])
        self.assertEqual(edit["reason"], "ambiguous_deletion_not_resolved_by_construct_role")


if __name__ == "__main__":
    unittest.main()

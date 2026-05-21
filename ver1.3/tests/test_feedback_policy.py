from __future__ import annotations

import unittest

from jp_speech_eval.feedback_policy import choose_feedback


class FeedbackPolicyTests(unittest.TestCase):
    def test_low_reliability_suppresses_metric_pileup(self) -> None:
        decision = choose_feedback(
            raw_feedback=[
                "这次录音里有些地方不够清楚，重录一次会更准。",
                "整体音高和示范音差得比较明显。",
                "句末语调和示范音不太一致。",
                "音高起伏较大，语气可能偏紧张或夸张。",
            ],
            reliability={"level": "low", "overall": 0.30, "f0_coverage": 0.3},
            mora_evidence_summary={"mora_count": 8, "judgement_available_count": 1},
        )
        self.assertEqual(decision.feedback, ["这次录音里有些地方不够清楚，重录一次会更准。"])
        self.assertGreaterEqual(len(decision.suppressed), 2)

    def test_actionable_prosody_beats_generic_summary(self) -> None:
        decision = choose_feedback(
            raw_feedback=[
                "整体音高和示范音差得比较明显。",
                "在「ア〜メ」附近，音高下降还不够明显。",
                "音高起伏较大，语气可能偏紧张或夸张。",
            ],
            reliability={"level": "high", "overall": 0.86, "f0_coverage": 0.9},
            mora_evidence_summary={"mora_count": 6, "judgement_available_count": 6},
        )
        self.assertIn("在「ア〜メ」附近，音高下降还不够明显。", decision.feedback)
        self.assertNotIn("整体音高和示范音差得比较明显。", decision.feedback)

    def test_tone_does_not_crowd_out_pronunciation(self) -> None:
        decision = choose_feedback(
            raw_feedback=[
                "节奏不太稳定，可能有拖音或卡顿。",
                "音量偏小，可能影响对方听清。",
            ],
            reliability={"level": "high", "overall": 0.88, "f0_coverage": 0.8},
            mora_evidence_summary={"mora_count": 8, "judgement_available_count": 7},
        )
        self.assertEqual(decision.feedback, ["节奏不太稳定，可能有拖音或卡顿。"])


if __name__ == "__main__":
    unittest.main()

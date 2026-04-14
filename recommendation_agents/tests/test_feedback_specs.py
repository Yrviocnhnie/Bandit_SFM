from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from recommendation_agents.feedback_specs import parse_phase2_feedback_markdown


class FeedbackSpecParserTest(unittest.TestCase):
    def test_parse_phase2_feedback_markdown_returns_two_feedbacks_per_context(self) -> None:
        markdown = "\n".join(
            [
                "# Phase 2",
                "",
                "## 1. SCENARIO: ARRIVE_OFFICE",
                "",
                "### Context A: Test context",
                "",
                "**Features**",
                "- state_current: office_arriving",
                "- precondition: commuting_walk_out",
                "- ps_time: morning",
                "",
                "**Simulated feedback**",
                "- like: O_SHOW_SCHEDULE",
                "- dislike: R_PLAN_DAY_OVER_COFFEE",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "phase2.md"
            path.write_text(markdown)
            items = parse_phase2_feedback_markdown(path)

        self.assertEqual(len(items), 2)
        self.assertEqual({item["feedback_type"] for item in items}, {"like", "dislike"})
        self.assertEqual({item["anchor_id"] for item in items}, {"arrive_office__context_a"})
        self.assertEqual(items[0]["scenario_id"], "ARRIVE_OFFICE")
        self.assertEqual(items[0]["anchor_context"]["state_current"], "office_arriving")
        self.assertIn("hour", items[0]["anchor_context"])

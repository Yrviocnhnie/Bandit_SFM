from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.catalog import build_ro_metadata_from_catalog_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]


class CatalogTest(unittest.TestCase):
    def test_build_ro_metadata_from_markdown(self) -> None:
        markdown_path = REPO_ROOT / "docs" / "scenario_recommendation_actions_v4.md"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "ro_metadata.json"
            summary = build_ro_metadata_from_catalog_markdown(markdown_path, output_path)
            self.assertEqual(summary.total_scenarios, 65)
            self.assertEqual(summary.total_actions, 167)
            self.assertEqual(summary.min_actions_per_scenario, 2)
            self.assertEqual(summary.max_actions_per_scenario, 3)

            payload = json.loads(output_path.read_text())
            arrive_office = next(item for item in payload["scenarios"] if item["scenario_id"] == "ARRIVE_OFFICE")
            self.assertEqual(
                [item["action_id"] for item in arrive_office["actions"]],
                ["arrive_office_coffee", "arrive_office_schedule", "arrive_office_todo_list"],
            )
            self.assertEqual(arrive_office["default_action_id"], "arrive_office_schedule")
            self.assertEqual(arrive_office["default_arm_id"], "ARRIVE_OFFICE::arrive_office_schedule")


if __name__ == "__main__":
    unittest.main()

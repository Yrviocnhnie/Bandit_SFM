from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.catalog import build_app_metadata_from_catalog_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]


class AppCatalogTest(unittest.TestCase):
    def test_build_app_metadata_from_markdown(self) -> None:
        markdown_path = REPO_ROOT / "docs" / "scenario_recommendation_actions_v4.md"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "app_metadata.json"
            summary = build_app_metadata_from_catalog_markdown(markdown_path, output_path)
            self.assertEqual(summary.total_scenarios, 65)
            self.assertEqual(summary.total_actions, 10)

            payload = json.loads(output_path.read_text())
            arrive_office = next(item for item in payload["scenarios"] if item["scenario_id"] == "ARRIVE_OFFICE")
            self.assertEqual(arrive_office["default_action_id"], "productivity")
            self.assertEqual(len(arrive_office["action_ids"]), 10)

            home_evening = next(item for item in payload["scenarios"] if item["scenario_id"] == "HOME_EVENING")
            self.assertEqual(home_evening["default_action_id"], "entertainment")


if __name__ == "__main__":
    unittest.main()

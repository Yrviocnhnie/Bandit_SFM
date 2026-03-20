from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.raw_synthetic import convert_raw_sequence_to_v0_app


REPO_ROOT = Path(__file__).resolve().parents[1]


class RawSyntheticAppConversionTest(unittest.TestCase):
    def test_convert_colleague_sequence_file_for_app_agent(self) -> None:
        input_path = REPO_ROOT / "docs" / "synthetic_bandit_v0_two_scenarios.jsonl"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_samples = Path(tmp_dir) / "converted_app.jsonl"
            output_metadata = Path(tmp_dir) / "app_metadata.json"
            summary = convert_raw_sequence_to_v0_app(
                input_path=input_path,
                output_samples_path=output_samples,
                output_metadata_path=output_metadata,
            )

            self.assertEqual(summary.input_rows, 2308)
            self.assertEqual(summary.kept_rows, 4)
            self.assertEqual(summary.unique_scenarios, 2)
            self.assertEqual(summary.unique_actions, 1)

            converted_rows = [json.loads(line) for line in output_samples.read_text().splitlines()]
            self.assertEqual(len(converted_rows), 4)
            self.assertEqual(converted_rows[0]["selected_action"], "productivity")

            metadata = json.loads(output_metadata.read_text())
            self.assertEqual(len(metadata["scenarios"]), 2)
            self.assertEqual(metadata["scenarios"][0]["default_action_id"], "productivity")

    def test_convert_finalized_first_step_file_for_app_agent(self) -> None:
        input_path = REPO_ROOT / "docs" / "synthetic_bandit_v0_two_scenarios_firststep_no_triggers.jsonl"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_samples = Path(tmp_dir) / "converted_app.jsonl"
            output_metadata = Path(tmp_dir) / "app_metadata.json"
            summary = convert_raw_sequence_to_v0_app(
                input_path=input_path,
                output_samples_path=output_samples,
                output_metadata_path=output_metadata,
            )

            self.assertEqual(summary.input_rows, 4)
            self.assertEqual(summary.kept_rows, 4)
            self.assertEqual(summary.unique_scenarios, 2)
            self.assertEqual(summary.unique_actions, 4)

            converted_rows = [json.loads(line) for line in output_samples.read_text().splitlines()]
            self.assertEqual(len(converted_rows), 4)
            self.assertEqual(converted_rows[0]["selected_action"], "productivity")
            self.assertEqual(converted_rows[1]["selected_action"], "music")

            metadata = json.loads(output_metadata.read_text())
            self.assertEqual(len(metadata["scenarios"]), 2)
            office_lunch_out = next(item for item in metadata["scenarios"] if item["scenario_id"] == "OFFICE_LUNCH_OUT")
            self.assertEqual(
                office_lunch_out["action_ids"],
                ["music", "navigation", "productivity", "shopping"],
            )


if __name__ == "__main__":
    unittest.main()

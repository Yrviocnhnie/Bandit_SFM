from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.raw_synthetic import convert_raw_sequence_to_v0


REPO_ROOT = Path(__file__).resolve().parents[1]


class RawSyntheticConversionTest(unittest.TestCase):
    def test_convert_colleague_sequence_file(self) -> None:
        input_path = REPO_ROOT / "docs" / "synthetic_bandit_v0_two_scenarios.jsonl"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_samples = Path(tmp_dir) / "converted.jsonl"
            output_metadata = Path(tmp_dir) / "metadata.json"
            summary = convert_raw_sequence_to_v0(
                input_path=input_path,
                output_samples_path=output_samples,
                output_metadata_path=output_metadata,
            )

            self.assertEqual(summary.input_rows, 2308)
            self.assertEqual(summary.kept_rows, 4)
            self.assertEqual(summary.unique_scenarios, 2)
            self.assertEqual(summary.unique_actions, 2)

            converted_rows = [json.loads(line) for line in output_samples.read_text().splitlines()]
            self.assertEqual(len(converted_rows), 4)
            self.assertEqual(converted_rows[0]["scenario_id"], "ARRIVE_OFFICE")
            self.assertEqual(converted_rows[0]["selected_action"], "arrive_office_schedule")
            self.assertEqual(converted_rows[0]["context"]["user_id_hash_bucket"], "b07")

            metadata = json.loads(output_metadata.read_text())
            self.assertEqual(len(metadata["scenarios"]), 2)
            self.assertEqual(metadata["scenarios"][0]["action_ids"], ["arrive_office_schedule"])

    def test_convert_finalized_first_step_file(self) -> None:
        input_path = REPO_ROOT / "docs" / "synthetic_bandit_v0_two_scenarios_firststep_no_triggers.jsonl"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_samples = Path(tmp_dir) / "converted.jsonl"
            output_metadata = Path(tmp_dir) / "metadata.json"
            summary = convert_raw_sequence_to_v0(
                input_path=input_path,
                output_samples_path=output_samples,
                output_metadata_path=output_metadata,
            )

            self.assertEqual(summary.input_rows, 4)
            self.assertEqual(summary.kept_rows, 4)
            self.assertEqual(summary.unique_scenarios, 2)
            self.assertEqual(summary.unique_actions, 4)

            converted_rows = [json.loads(line) for line in output_samples.read_text().splitlines()]
            self.assertEqual(converted_rows[0]["scenario_id"], "ARRIVE_OFFICE")
            self.assertEqual(converted_rows[0]["selected_action"], "arrive_office_schedule")
            self.assertEqual(converted_rows[0]["event_id"], "arrive_office_ep01:0")
            self.assertEqual(converted_rows[0]["context"]["user_id_hash_bucket"], "b07")

            metadata = json.loads(output_metadata.read_text())
            self.assertEqual(len(metadata["scenarios"]), 2)
            arrive_office = next(item for item in metadata["scenarios"] if item["scenario_id"] == "ARRIVE_OFFICE")
            self.assertEqual(
                arrive_office["action_ids"],
                ["arrive_office_coffee", "arrive_office_schedule"],
            )


if __name__ == "__main__":
    unittest.main()

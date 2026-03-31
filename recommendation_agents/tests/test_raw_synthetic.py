from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.raw_synthetic import convert_raw_sequence_to_v6_expanded_ro


def _context() -> dict[str, object]:
    return {
        "state_current": "office_arriving",
        "precondition": "commuting_walk_out",
        "state_duration_sec": 180,
        "ps_time": "morning",
        "hour": 9,
        "cal_hasUpcoming": 1,
        "ps_dayType": "workday",
        "ps_motion": "stationary",
        "wifiLost": 0,
        "wifiLostCategory": "work",
        "cal_eventCount": 2,
        "cal_inMeeting": 0,
        "cal_nextLocation": "work",
        "ps_sound": "quiet",
        "sms_delivery_pending": 0,
        "sms_train_pending": 0,
        "sms_flight_pending": 0,
        "sms_hotel_pending": 0,
        "sms_movie_pending": 0,
        "sms_hospital_pending": 0,
        "sms_ride_pending": 0,
        "timestep": 32400,
        "ps_location": "work",
        "ps_phone": "on_desk",
        "batteryLevel": 88,
        "isCharging": 1,
        "networkType": "wifi",
        "activityState": "sitting",
        "activityDuration": 900,
        "user_id_hash_bucket": "b07",
        "age_bucket": "25_34",
        "sex": "female",
        "has_kids": 0,
    }


class RawSyntheticExpansionTest(unittest.TestCase):
    def _write_fixture(self, tmp_path: Path) -> tuple[Path, Path]:
        raw_path = tmp_path / "raw.jsonl"
        relevance_path = tmp_path / "v6.md"
        raw_rows = [
            {
                "episode_id": "arrive_office_ep01",
                "scenario_id": "ARRIVE_OFFICE",
                "scenario_elapsed_sec": 0,
                "emit_recommendation": 1,
                "gt_ro": "O_SHOW_SCHEDULE",
                "features": _context(),
            }
        ]
        raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
        relevance_path.write_text(
            "\n".join(
                [
                    "# v6",
                    "",
                    "## Scenario Defaults",
                    "",
                    "| scenarioId | scenarioNameZh | most relevant 3 R/O actionIds | other plausible 3 R/O actionIds | irrelevant 2 R/O actionIds | most relevant 3 app categories | other plausible 3 app categories | irrelevant 2 app categories |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- |",
                    "| `ARRIVE_OFFICE` | 到达办公室 | `A1`<br>`A2`<br>`A3` | `A4`<br>`A5`<br>`A6` | `A7`<br>`A8` | `productivity`<br>`news`<br>`music` | `reading`<br>`social`<br>`health` | `shopping`<br>`game` |",
                ]
            )
        )
        return raw_path, relevance_path

    def test_exclude_most_plausible_expands_remaining_actions_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_path, relevance_path = self._write_fixture(tmp_path)
            output_path = tmp_path / "expanded.jsonl"

            summary = convert_raw_sequence_to_v6_expanded_ro(
                input_path=raw_path,
                output_samples_path=output_path,
                relevance_markdown=relevance_path,
                most_relevant_reward=1.0,
                plausible_reward=0.1,
                irrelevant_reward=0.0,
                most_relevant_repeat=1,
                plausible_repeat=1,
                irrelevant_repeat=0,
                all_action_ids=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
                other_zero_mode="exclude-most-plausible",
            )

            self.assertEqual(summary.emitted_samples, 8)
            self.assertEqual(summary.tier_counts["most_relevant"], 3)
            self.assertEqual(summary.tier_counts["plausible"], 3)
            self.assertEqual(summary.tier_counts["other_zero"], 2)

    def test_exclude_most_only_expands_all_non_most_actions_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_path, relevance_path = self._write_fixture(tmp_path)
            output_path = tmp_path / "expanded.jsonl"

            summary = convert_raw_sequence_to_v6_expanded_ro(
                input_path=raw_path,
                output_samples_path=output_path,
                relevance_markdown=relevance_path,
                most_relevant_reward=1.0,
                plausible_reward=0.0,
                irrelevant_reward=0.0,
                most_relevant_repeat=1,
                plausible_repeat=0,
                irrelevant_repeat=0,
                all_action_ids=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
                other_zero_mode="exclude-most-only",
            )

            self.assertEqual(summary.emitted_samples, 8)
            self.assertEqual(summary.tier_counts["most_relevant"], 3)
            self.assertEqual(summary.tier_counts["other_zero"], 5)

    def test_exclude_all_labeled_keeps_irrelevant_separate_from_zero_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_path, relevance_path = self._write_fixture(tmp_path)
            output_path = tmp_path / "expanded.jsonl"

            summary = convert_raw_sequence_to_v6_expanded_ro(
                input_path=raw_path,
                output_samples_path=output_path,
                relevance_markdown=relevance_path,
                most_relevant_reward=1.0,
                plausible_reward=0.1,
                irrelevant_reward=-0.1,
                most_relevant_repeat=1,
                plausible_repeat=1,
                irrelevant_repeat=1,
                all_action_ids=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"],
                other_zero_mode="exclude-all-labeled",
            )

            self.assertEqual(summary.emitted_samples, 10)
            self.assertEqual(summary.tier_counts["most_relevant"], 3)
            self.assertEqual(summary.tier_counts["plausible"], 3)
            self.assertEqual(summary.tier_counts["irrelevant"], 2)
            self.assertEqual(summary.tier_counts["other_zero"], 2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.trainer import train_v0
from recommendation_agents.workflows import evaluate_v0_topk


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _context(**overrides):
    payload = {
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
    payload.update(overrides)
    return payload


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for the workflow evaluation tests")
class WorkflowEvaluationTest(unittest.TestCase):
    def test_evaluate_v0_topk_reports_relevance_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metadata_path = tmp_path / "metadata.json"
            samples_path = tmp_path / "samples.jsonl"
            artifact_dir = tmp_path / "artifact"
            catalog_path = tmp_path / "scenario_recommendation_actions_v6.md"

            metadata = {
                "schema_version": "ro-global-catalog",
                "global_action_ids": [
                    "O_SHOW_SCHEDULE",
                    "O_SHOW_TODO",
                    "R_PLAN_DAY_OVER_COFFEE",
                    "O_SHOW_NEARBY_OPTIONS",
                ],
                "actions": [
                    {"action_id": "O_SHOW_SCHEDULE", "display_name": "Show schedule"},
                    {"action_id": "O_SHOW_TODO", "display_name": "Show todo"},
                    {"action_id": "R_PLAN_DAY_OVER_COFFEE", "display_name": "Plan day over coffee"},
                    {"action_id": "O_SHOW_NEARBY_OPTIONS", "display_name": "Show nearby options"},
                ],
                "scenario_default_rankings": [
                    {
                        "scenario_id": "ARRIVE_OFFICE",
                        "default_action_ids": [
                            "O_SHOW_SCHEDULE",
                            "O_SHOW_TODO",
                            "R_PLAN_DAY_OVER_COFFEE",
                        ],
                    },
                    {
                        "scenario_id": "HOME_EVENING",
                        "default_action_ids": [
                            "R_PLAN_DAY_OVER_COFFEE",
                            "O_SHOW_TODO",
                            "O_SHOW_NEARBY_OPTIONS",
                        ],
                    },
                ],
            }
            samples = [
                {
                    "event_id": "evt-001",
                    "scenario_id": "ARRIVE_OFFICE",
                    "context": _context(),
                    "selected_action": "O_SHOW_SCHEDULE",
                    "reward": 1.0,
                    "propensity": 1.0,
                },
                {
                    "event_id": "evt-002",
                    "scenario_id": "ARRIVE_OFFICE",
                    "context": _context(hour=8, cal_eventCount=4),
                    "selected_action": "O_SHOW_TODO",
                    "reward": 0.7,
                    "propensity": 1.0,
                },
                {
                    "event_id": "evt-003",
                    "scenario_id": "HOME_EVENING",
                    "context": _context(
                        state_current="home_evening",
                        precondition="office_working",
                        ps_time="evening",
                        hour=19,
                        ps_location="home",
                        wifiLostCategory="home",
                    ),
                    "selected_action": "R_PLAN_DAY_OVER_COFFEE",
                    "reward": 1.0,
                    "propensity": 1.0,
                },
                {
                    "event_id": "evt-004",
                    "scenario_id": "HOME_EVENING",
                    "context": _context(
                        state_current="home_evening",
                        precondition="office_working",
                        ps_time="evening",
                        hour=20,
                        ps_location="home",
                        wifiLostCategory="home",
                    ),
                    "selected_action": "O_SHOW_NEARBY_OPTIONS",
                    "reward": 0.5,
                    "propensity": 1.0,
                },
            ]

            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
            samples_path.write_text("".join(json.dumps(row) + "\n" for row in samples))
            catalog_path.write_text(
                "\n".join(
                    [
                        "# v6",
                        "",
                        "## Scenario Defaults",
                        "",
                        "| scenarioId | scenarioNameZh | most relevant 3 R/O actionIds | other plausible 3 R/O actionIds | irrelevant 2 R/O actionIds | most relevant 3 app categories | other plausible 3 app categories | irrelevant 2 app categories |",
                        "| --- | --- | --- | --- | --- | --- | --- | --- |",
                        "| `ARRIVE_OFFICE` | 到达办公室 | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODO`<br>`R_PLAN_DAY_OVER_COFFEE` | `O_SHOW_NEARBY_OPTIONS`<br>`O_SHOW_SCHEDULE`<br>`O_SHOW_TODO` | `R_PLAN_DAY_OVER_COFFEE`<br>`O_SHOW_NEARBY_OPTIONS` | `productivity`<br>`news`<br>`music` | `reading`<br>`social`<br>`health` | `shopping`<br>`game` |",
                        "| `HOME_EVENING` | 居家傍晚 | `R_PLAN_DAY_OVER_COFFEE`<br>`O_SHOW_TODO`<br>`O_SHOW_NEARBY_OPTIONS` | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODO`<br>`R_PLAN_DAY_OVER_COFFEE` | `O_SHOW_SCHEDULE`<br>`O_SHOW_NEARBY_OPTIONS` | `music`<br>`reading`<br>`health` | `social`<br>`news`<br>`productivity` | `shopping`<br>`game` |",
                    ]
                )
            )

            train_v0(
                metadata_path=metadata_path,
                samples_path=samples_path,
                output_dir=artifact_dir,
                alpha=0.05,
                default_bonus=0.75,
                device="cpu",
                progress_every=10,
            )

            summary = evaluate_v0_topk(
                artifact_dir=artifact_dir,
                metadata_path=metadata_path,
                test_samples_path=samples_path,
                catalog_markdown=catalog_path,
                label_namespace="ro",
                top_k=3,
                device="cpu",
                progress_every=10,
            )

            self.assertEqual(summary.sample_count, 4)
            self.assertGreaterEqual(summary.avg_most_relevant_covered_in_topk, 0.0)
            self.assertGreaterEqual(summary.avg_acceptable_covered_in_topk, 0.0)
            self.assertGreaterEqual(summary.avg_irrelevant_in_topk, 0.0)
            self.assertEqual(summary.scenarios_with_test_samples, 2)
            self.assertEqual(summary.scenarios_without_test_samples, [])
            self.assertEqual(set(summary.per_scenario), {"ARRIVE_OFFICE", "HOME_EVENING"})
            self.assertEqual(
                summary.per_scenario["ARRIVE_OFFICE"]["most_relevant_3"],
                ["O_SHOW_SCHEDULE", "O_SHOW_TODO", "R_PLAN_DAY_OVER_COFFEE"],
            )
            self.assertEqual(len(summary.per_scenario["ARRIVE_OFFICE"]["other_plausible_3"]), 3)
            self.assertEqual(len(summary.per_scenario["ARRIVE_OFFICE"]["irrelevant_2"]), 2)
            self.assertLessEqual(
                len(summary.per_scenario["ARRIVE_OFFICE"]["predicted_top6_action_distribution"]),
                6,
            )
            self.assertLessEqual(len(summary.top6_predicted_action_distribution), 6)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.trainer import train_v0
from recommendation_agents.workflows import (
    _select_hard_negative_candidates,
    eval_soft_scenarios_both,
    evaluate_v0_topk,
    render_hard_negative_candidates_markdown,
    run_v6_plan_a,
    run_v6_plan_all_data,
    run_v6_plan_b,
    split_raw_by_scenario_episode,
    HardNegativeMiningSummary,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


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


class HardNegativeMiningHelpersTest(unittest.TestCase):
    def test_select_hard_negative_candidates_sorts_and_filters(self) -> None:
        candidates = _select_hard_negative_candidates(
            Counter(
                {
                    "O_SHOW_WEATHER": 9,
                    "O_SHOW_SCHEDULE": 7,
                    "O_SHOW_NEARBY_OPTIONS": 3,
                }
            ),
            sample_count=10,
            exclude_action_ids={"O_SHOW_SCHEDULE"},
            max_candidates=2,
            min_count=2,
            min_rate=0.2,
        )
        self.assertEqual(
            [row["action_id"] for row in candidates],
            ["O_SHOW_WEATHER", "O_SHOW_NEARBY_OPTIONS"],
        )
        self.assertAlmostEqual(candidates[0]["rate"], 0.9)

    def test_render_hard_negative_candidates_markdown_includes_candidates(self) -> None:
        summary = HardNegativeMiningSummary(
            raw_input_path="train.raw.jsonl",
            artifact_dir="artifact",
            metadata_path="metadata.json",
            catalog_markdown="docs/scenario_recommendation_actions_v6.md",
            label_namespace="ro",
            top_k=6,
            sample_count=12,
            scenarios_with_samples=1,
            max_candidates_per_scenario=3,
            per_scenario={
                "ARRIVE_OFFICE": {
                    "sample_count": 12,
                    "most_relevant_3": ["O_SHOW_SCHEDULE", "O_SHOW_TODO", "R_PLAN_DAY_OVER_COFFEE"],
                    "other_plausible_3": ["O_SHOW_WEATHER", "O_SHOW_NEARBY_OPTIONS", "O_SHOW_NEWS"],
                    "irrelevant_2": ["O_SHOW_PAYMENT_QR", "O_SHOW_RESTAURANTS"],
                    "candidate_hard_negatives": [
                        {"action_id": "O_SHOW_BOOKING_DETAILS", "display_action_id": "O_SHOW_BOOKING_DETAILS", "count": 4, "rate": 0.3333}
                    ],
                    "top10_non_acceptable_predicted_actions": [],
                }
            },
        )
        markdown = render_hard_negative_candidates_markdown(summary)
        self.assertIn("ARRIVE_OFFICE", markdown)
        self.assertIn("O_SHOW_BOOKING_DETAILS", markdown)


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

    def test_eval_soft_scenarios_both_reports_top3_and_top5_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_dir = tmp_path / "prepared"
            data_dir.mkdir()
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
            }
            app_metadata = {
                "schema_version": "app-global-catalog",
                "global_action_ids": ["productivity", "news", "music", "reading", "social"],
                "actions": [
                    {"action_id": action_id, "display_name": action_id}
                    for action_id in ["productivity", "news", "music", "reading", "social"]
                ],
            }
            ro_train_rows = [
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
            ]
            app_train_rows = [
                {
                    "event_id": "app-001",
                    "scenario_id": "ARRIVE_OFFICE",
                    "context": _context(),
                    "selected_action": "productivity",
                    "reward": 1.0,
                    "propensity": 1.0,
                },
                {
                    "event_id": "app-002",
                    "scenario_id": "ARRIVE_OFFICE",
                    "context": _context(hour=8, cal_eventCount=4),
                    "selected_action": "news",
                    "reward": 0.7,
                    "propensity": 1.0,
                },
            ]
            soft_raw_rows = [
                {
                    "scenario_id": "RETURN_OFFICE_AFTER_COFFEE",
                    "scenario_name": "Return to office after coffee",
                    "sample_id": "soft-001",
                    "features": _context(precondition="at_cafe", state_current="office_arriving", ps_time="morning", hour=9),
                    "gt_ro": "O_SHOW_SCHEDULE",
                    "gt_app": "productivity",
                },
                {
                    "scenario_id": "RETURN_OFFICE_AFTER_COFFEE",
                    "scenario_name": "Return to office after coffee",
                    "sample_id": "soft-002",
                    "features": _context(precondition="at_cafe_quiet", state_current="office_working", ps_time="forenoon", hour=10),
                    "gt_ro": "O_SHOW_TODO",
                    "gt_app": "news",
                },
            ]
            spec_path = tmp_path / "soft_scenarios.md"
            raw_path = tmp_path / "soft.jsonl"

            (data_dir / "ro_metadata.json").write_text(json.dumps(metadata, indent=2))
            (data_dir / "app_metadata.json").write_text(json.dumps(app_metadata, indent=2))
            (data_dir / "ro_train_samples_expanded.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ro_train_rows))
            (data_dir / "app_train_samples_expanded.jsonl").write_text("".join(json.dumps(row) + "\n" for row in app_train_rows))
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in soft_raw_rows))
            spec_path.write_text(
                "\n".join(
                    [
                        "# In-Between",
                        "",
                        "## RETURN_OFFICE_AFTER_COFFEE — Return to office after coffee",
                        "- **Expanded RO ranking (top 10):** `O_SHOW_TODO`, `O_SHOW_SCHEDULE`, `R_PLAN_DAY_OVER_COFFEE`, `O_SHOW_NEARBY_OPTIONS`, `O_SHOW_TODO`, `O_SHOW_SCHEDULE`, `R_PLAN_DAY_OVER_COFFEE`, `O_SHOW_NEARBY_OPTIONS`, `O_SHOW_TODO`, `O_SHOW_SCHEDULE`",
                        "- **Expanded App ranking (top 5):** `productivity`, `news`, `music`, `reading`, `social`",
                    ]
                )
            )

            train_v0(
                metadata_path=data_dir / "ro_metadata.json",
                samples_path=data_dir / "ro_train_samples_expanded.jsonl",
                output_dir=data_dir / "ro_model",
                alpha=0.05,
                default_bonus=0.0,
                device="cpu",
                progress_every=10,
            )
            train_v0(
                metadata_path=data_dir / "app_metadata.json",
                samples_path=data_dir / "app_train_samples_expanded.jsonl",
                output_dir=data_dir / "app_model",
                alpha=0.05,
                default_bonus=0.0,
                device="cpu",
                progress_every=10,
            )

            summary = eval_soft_scenarios_both(
                data_dir=data_dir,
                raw_input_path=raw_path,
                spec_markdown=spec_path,
                device="cpu",
                progress_every=10,
            )

            self.assertEqual(summary.ro.sample_count, 2)
            self.assertEqual(summary.app.sample_count, 2)
            self.assertGreaterEqual(summary.ro.avg_hits_in_top3, 0.0)
            self.assertGreaterEqual(summary.ro.avg_hits_in_top5, summary.ro.avg_hits_in_top3)
            self.assertGreaterEqual(summary.app.avg_hits_in_top3, 0.0)
            self.assertGreaterEqual(summary.app.avg_hits_in_top5, summary.app.avg_hits_in_top3)
            self.assertEqual(summary.ro.scenarios_with_samples, 1)
            self.assertEqual(summary.app.scenarios_with_samples, 1)
            self.assertIn("RETURN_OFFICE_AFTER_COFFEE", summary.ro.per_scenario)
            self.assertIn("RETURN_OFFICE_AFTER_COFFEE", summary.app.per_scenario)
            self.assertTrue((data_dir / "eval_soft_scenarios.json").exists())

    def test_run_v6_plan_a_reuses_existing_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "existing_prepared"
            output_dir = tmp_path / "plan_a"
            relevance_path = tmp_path / "plan_a_v6.md"
            input_dir.mkdir()
            metadata = {
                "schema_version": "ro-global-catalog",
                "global_action_ids": [
                    "O_SHOW_SCHEDULE",
                    "O_SHOW_TODO",
                    "R_PLAN_DAY_OVER_COFFEE",
                    "O_SHOW_NEARBY_OPTIONS",
                    "O_SHOW_BOOKING_DETAILS",
                    "R_HOME_RELAX_AND_RESET",
                    "O_SHOW_WEATHER",
                ],
                "actions": [
                    {"action_id": "O_SHOW_SCHEDULE", "display_name": "Show schedule"},
                    {"action_id": "O_SHOW_TODO", "display_name": "Show todo"},
                    {"action_id": "R_PLAN_DAY_OVER_COFFEE", "display_name": "Plan day over coffee"},
                    {"action_id": "O_SHOW_NEARBY_OPTIONS", "display_name": "Show nearby"},
                    {"action_id": "O_SHOW_BOOKING_DETAILS", "display_name": "Show booking"},
                    {"action_id": "R_HOME_RELAX_AND_RESET", "display_name": "Relax at home"},
                    {"action_id": "O_SHOW_WEATHER", "display_name": "Show weather"},
                ],
            }
            app_metadata = {
                "schema_version": "app-global-catalog",
                "global_action_ids": ["productivity", "news", "music", "reading", "social", "health", "shopping", "game"],
                "actions": [{"action_id": action_id, "display_name": action_id} for action_id in ["productivity", "news", "music", "reading", "social", "health", "shopping", "game"]],
            }
            train_rows = [
                {
                    "episode_id": "arrive_office_ep01",
                    "scenario_id": "ARRIVE_OFFICE",
                    "scenario_elapsed_sec": 0,
                    "emit_recommendation": 1,
                    "gt_ro": "O_SHOW_SCHEDULE",
                    "gt_app": "productivity",
                    "features": _context(),
                },
                {
                    "episode_id": "home_evening_ep01",
                    "scenario_id": "HOME_EVENING",
                    "scenario_elapsed_sec": 0,
                    "emit_recommendation": 1,
                    "gt_ro": "R_PLAN_DAY_OVER_COFFEE",
                    "gt_app": "music",
                    "features": _context(
                        state_current="home_evening",
                        precondition="office_working",
                        ps_time="evening",
                        hour=19,
                        ps_location="home",
                        wifiLostCategory="home",
                    ),
                },
            ]
            test_rows = [
                {
                    "event_id": "test-arrive",
                    "scenario_id": "ARRIVE_OFFICE",
                    "context": _context(hour=8),
                    "selected_action": "O_SHOW_SCHEDULE",
                    "reward": 1.0,
                    "propensity": 1.0,
                }
            ]
            app_test_rows = [
                {
                    "event_id": "test-arrive-app",
                    "scenario_id": "ARRIVE_OFFICE",
                    "context": _context(hour=8),
                    "selected_action": "productivity",
                    "reward": 1.0,
                    "propensity": 1.0,
                }
            ]
            (input_dir / "train.raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_rows))
            (input_dir / "test.raw.jsonl").write_text("".join(json.dumps(train_rows[0]) + "\n"))
            (input_dir / "ro_metadata.json").write_text(json.dumps(metadata, indent=2))
            (input_dir / "app_metadata.json").write_text(json.dumps(app_metadata, indent=2))
            (input_dir / "ro_test_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in test_rows))
            (input_dir / "app_test_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in app_test_rows))
            relevance_path.write_text(
                "\n".join(
                    [
                        "# v6",
                        "",
                        "## Scenario Defaults",
                        "",
                        "| scenarioId | scenarioNameZh | most relevant 3 R/O actionIds | other plausible 3 R/O actionIds | irrelevant 2 R/O actionIds | most relevant 3 app categories | other plausible 3 app categories | irrelevant 2 app categories |",
                        "| --- | --- | --- | --- | --- | --- | --- | --- |",
                        "| `ARRIVE_OFFICE` | 到达办公室 | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODO`<br>`R_PLAN_DAY_OVER_COFFEE` | `O_SHOW_NEARBY_OPTIONS`<br>`O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_WEATHER` | `R_HOME_RELAX_AND_RESET`<br>`O_SHOW_WEATHER` | `productivity`<br>`news`<br>`music` | `reading`<br>`social`<br>`health` | `shopping`<br>`game` |",
                        "| `HOME_EVENING` | 居家傍晚 | `R_PLAN_DAY_OVER_COFFEE`<br>`R_HOME_RELAX_AND_RESET`<br>`O_SHOW_WEATHER` | `O_SHOW_TODO`<br>`O_SHOW_SCHEDULE`<br>`O_SHOW_NEARBY_OPTIONS` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_SCHEDULE` | `music`<br>`reading`<br>`health` | `social`<br>`news`<br>`productivity` | `shopping`<br>`game` |",
                    ]
                )
            )

            summary = run_v6_plan_a(
                input_data_dir=input_dir,
                output_dir=output_dir,
                relevance_markdown=relevance_path,
                alpha=0.05,
                default_bonus=0.0,
                device="cpu",
                progress_every=10,
            )

            self.assertEqual(summary.ro_expansion["emitted_samples"], 16)
            self.assertEqual(summary.app_expansion["emitted_samples"], 16)
            self.assertEqual(
                (output_dir / "ro_test_samples.jsonl").read_text(),
                (input_dir / "ro_test_samples.jsonl").read_text(),
            )
            self.assertTrue((output_dir / "eval_both_top3.json").exists())

    def test_run_v6_plan_b_creates_stratified_split_and_trains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_path = tmp_path / "raw.jsonl"
            rows = []
            for index in range(6):
                rows.append(
                    {
                        "episode_id": f"arrive_office_ep{index:02d}",
                        "scenario_id": "ARRIVE_OFFICE",
                        "scenario_elapsed_sec": 0,
                        "emit_recommendation": 1,
                        "gt_ro": "O_SHOW_SCHEDULE",
                        "gt_app": "productivity",
                        "features": _context(hour=8 + index),
                    }
                )
                rows.append(
                    {
                        "episode_id": f"home_evening_ep{index:02d}",
                        "scenario_id": "HOME_EVENING",
                        "scenario_elapsed_sec": 0,
                        "emit_recommendation": 1,
                        "gt_ro": "R_PLAN_DAY_OVER_COFFEE",
                        "gt_app": "music",
                        "features": _context(
                            state_current="home_evening",
                            precondition="office_working",
                            ps_time="evening",
                            hour=19,
                            ps_location="home",
                            wifiLostCategory="home",
                            batteryLevel=70 + index,
                        ),
                    }
                )
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            summary = run_v6_plan_b(
                input_path=raw_path,
                catalog_markdown=ROOT / "docs/scenario_recommendation_actions_v5.md",
                output_dir=tmp_path / "plan_b",
                relevance_markdown=ROOT / "docs/scenario_recommendation_actions_v6.md",
                test_ratio=0.2,
                alpha=0.05,
                default_bonus=0.0,
                device="cpu",
                progress_every=10,
            )

            self.assertTrue((tmp_path / "plan_b" / "train.raw.jsonl").exists())
            self.assertTrue((tmp_path / "plan_b" / "test.raw.jsonl").exists())
            self.assertTrue((tmp_path / "plan_b" / "ro_train_samples_expanded.jsonl").exists())
            self.assertTrue((tmp_path / "plan_b" / "app_train_samples_expanded.jsonl").exists())
            self.assertTrue((tmp_path / "plan_b" / "eval_both_top3.json").exists())
            self.assertIn("split_summary", summary.extra_summary)

    def test_run_v6_plan_all_data_uses_full_raw_for_train_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_path = tmp_path / "raw.jsonl"
            rows = []
            for index in range(4):
                rows.append(
                    {
                        "episode_id": f"arrive_office_ep{index:02d}",
                        "scenario_id": "ARRIVE_OFFICE",
                        "scenario_elapsed_sec": 0,
                        "emit_recommendation": 1,
                        "gt_ro": "O_SHOW_SCHEDULE",
                        "gt_app": "productivity",
                        "features": _context(hour=8 + index),
                    }
                )
                rows.append(
                    {
                        "episode_id": f"home_evening_ep{index:02d}",
                        "scenario_id": "HOME_EVENING",
                        "scenario_elapsed_sec": 0,
                        "emit_recommendation": 1,
                        "gt_ro": "R_PLAN_DAY_OVER_COFFEE",
                        "gt_app": "music",
                        "features": _context(
                            state_current="home_evening",
                            precondition="office_working",
                            ps_time="evening",
                            hour=19,
                            ps_location="home",
                            wifiLostCategory="home",
                            batteryLevel=70 + index,
                        ),
                    }
                )
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            output_dir = tmp_path / "plan_all_data"
            summary = run_v6_plan_all_data(
                input_path=raw_path,
                catalog_markdown=ROOT / "docs/scenario_recommendation_actions_v5.md",
                output_dir=output_dir,
                relevance_markdown=ROOT / "docs/scenario_recommendation_actions_v6.md",
                alpha=0.05,
                default_bonus=0.0,
                device="cpu",
                progress_every=10,
            )

            self.assertTrue((output_dir / "train.raw.jsonl").exists())
            self.assertTrue((output_dir / "test.raw.jsonl").exists())
            self.assertEqual((output_dir / "train.raw.jsonl").read_text(), raw_path.read_text())
            self.assertEqual((output_dir / "test.raw.jsonl").read_text(), raw_path.read_text())
            self.assertTrue((output_dir / "ro_train_samples_expanded.jsonl").exists())
            self.assertTrue((output_dir / "app_train_samples_expanded.jsonl").exists())
            self.assertTrue((output_dir / "ro_test_samples.jsonl").exists())
            self.assertTrue((output_dir / "app_test_samples.jsonl").exists())
            self.assertTrue((output_dir / "eval_both_top3.json").exists())
            self.assertEqual(summary.extra_summary["split_policy"], "all_rows_for_train_and_test")
            self.assertEqual(summary.extra_summary["raw_row_count"], len(rows))


class StratifiedSplitTest(unittest.TestCase):
    def test_split_raw_by_scenario_episode_keeps_train_and_test_per_scenario_when_possible(self) -> None:
        rows = []
        for index in range(5):
            rows.append({"episode_id": f"a-{index}", "scenario_id": "ARRIVE_OFFICE", "scenario_elapsed_sec": 0})
            rows.append({"episode_id": f"h-{index}", "scenario_id": "HOME_EVENING", "scenario_elapsed_sec": 0})
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "raw.jsonl"
            train_path = tmp_path / "train.jsonl"
            test_path = tmp_path / "test.jsonl"
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            summary = split_raw_by_scenario_episode(input_path, train_path, test_path, test_ratio=0.2)

            self.assertEqual(summary.train_scenarios, 2)
            self.assertEqual(summary.test_scenarios, 2)
            self.assertGreater(summary.per_scenario["ARRIVE_OFFICE"]["train_rows"], 0)
            self.assertGreater(summary.per_scenario["ARRIVE_OFFICE"]["test_rows"], 0)
            self.assertGreater(summary.per_scenario["HOME_EVENING"]["train_rows"], 0)
            self.assertGreater(summary.per_scenario["HOME_EVENING"]["test_rows"], 0)


if __name__ == "__main__":
    unittest.main()

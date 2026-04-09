from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.raw_synthetic import convert_raw_sequence_to_v0_app, convert_raw_sequence_to_v6_expanded_app


def _base_features(**overrides):
    payload = {
        'state_current': 'office_arriving',
        'precondition': 'commuting_walk_out',
        'state_duration_sec': 180,
        'ps_time': 'morning',
        'hour': 9,
        'cal_hasUpcoming': 1,
        'ps_dayType': 'workday',
        'ps_motion': 'stationary',
        'wifiLost': 0,
        'wifiLostCategory': 'work',
        'cal_eventCount': 2,
        'cal_inMeeting': 0,
        'cal_nextLocation': 'work',
        'ps_sound': 'quiet',
        'sms_delivery_pending': 0,
        'sms_train_pending': 0,
        'sms_flight_pending': 0,
        'sms_hotel_pending': 0,
        'sms_movie_pending': 0,
        'sms_hospital_pending': 0,
        'sms_ride_pending': 0,
        'timestep': 32400,
        'ps_location': 'work',
        'ps_phone': 'on_desk',
        'batteryLevel': 88,
        'isCharging': 1,
        'networkType': 'wifi',
        'activityState': 'sitting',
        'activityDuration': 900,
        'user_id_hash_bucket': 'b07',
        'age_bucket': '25_34',
        'sex': 'female',
        'has_kids': 0,
    }
    payload.update(overrides)
    return payload


class RawSyntheticAppConversionTest(unittest.TestCase):
    def test_convert_raw_sequence_to_global_app_samples(self) -> None:
        rows = [
            {
                'episode_id': 'arrive_office_ep01',
                'scenario_id': 'ARRIVE_OFFICE',
                'scenario_elapsed_sec': 0,
                'emit_recommendation': 1,
                'gt_app': 'productivity',
                'features': _base_features(),
            },
            {
                'episode_id': 'arrive_office_ep01',
                'scenario_id': 'ARRIVE_OFFICE',
                'scenario_elapsed_sec': 30,
                'emit_recommendation': 1,
                'gt_app': 'music',
                'features': _base_features(hour=8),
            },
            {
                'episode_id': 'home_evening_ep01',
                'scenario_id': 'HOME_EVENING',
                'scenario_elapsed_sec': 0,
                'emit_recommendation': 1,
                'gt_app': 'entertainment',
                'features': _base_features(state_current='home_evening', precondition='office_working', ps_time='evening', hour=19, ps_location='home', wifiLostCategory='home'),
            },
            {
                'episode_id': 'noop_ep01',
                'scenario_id': 'ARRIVE_OFFICE',
                'scenario_elapsed_sec': 60,
                'emit_recommendation': 0,
                'gt_app': 'NONE',
                'features': _base_features(),
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / 'raw.jsonl'
            input_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            output_samples = tmp_path / 'converted_app.jsonl'
            output_metadata = tmp_path / 'app_metadata.json'

            summary = convert_raw_sequence_to_v0_app(input_path, output_samples, output_metadata)

            self.assertEqual(summary.input_rows, 4)
            self.assertEqual(summary.kept_rows, 3)
            self.assertEqual(summary.unique_scenarios, 2)
            self.assertEqual(summary.unique_actions, 3)

            converted_rows = [json.loads(line) for line in output_samples.read_text().splitlines()]
            self.assertEqual(converted_rows[0]['selected_action'], 'productivity')
            self.assertEqual(converted_rows[0]['context']['precondition'], 'commuting_walk_out')
            self.assertNotIn('transportMode', converted_rows[0]['context'])
            self.assertEqual(converted_rows[1]['selected_action'], 'music')

            metadata = json.loads(output_metadata.read_text())
            self.assertEqual(metadata['global_action_ids'], ['entertainment', 'music', 'productivity'])
            arrive_office = next(item for item in metadata['scenario_default_rankings'] if item['scenario_id'] == 'ARRIVE_OFFICE')
            self.assertEqual(arrive_office['default_action_ids'], ['productivity', 'music'])

    def test_convert_raw_sequence_to_v6_expanded_app_emits_8_rows_per_context(self) -> None:
        rows = [
            {
                'episode_id': 'arrive_office_ep01',
                'scenario_id': 'ARRIVE_OFFICE',
                'scenario_elapsed_sec': 0,
                'emit_recommendation': 1,
                'gt_app': 'productivity',
                'features': _base_features(),
            }
        ]
        markdown = "\n".join(
            [
                "# v6",
                "",
                "## Scenario Defaults",
                "",
                "| scenarioId | scenarioNameZh | most relevant 3 R/O actionIds | other plausible 3 R/O actionIds | irrelevant 2 R/O actionIds | most relevant 3 app categories | other plausible 3 app categories | irrelevant 2 app categories |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "| `ARRIVE_OFFICE` | 到达办公室 | `A1`<br>`A2`<br>`A3` | `A4`<br>`A5`<br>`A6` | `A7`<br>`A8` | `app1`<br>`app2`<br>`app3` | `app4`<br>`app5`<br>`app6` | `app7`<br>`app8` |",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / 'raw.jsonl'
            input_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            output_samples = tmp_path / 'expanded_app.jsonl'
            relevance_path = tmp_path / 'v6.md'
            relevance_path.write_text(markdown)

            summary = convert_raw_sequence_to_v6_expanded_app(
                input_path=input_path,
                output_samples_path=output_samples,
                relevance_markdown=relevance_path,
            )

            self.assertEqual(summary.input_rows, 1)
            self.assertEqual(summary.kept_rows, 1)
            self.assertEqual(summary.emitted_samples, 8)
            self.assertEqual(summary.tier_counts, {'most_relevant': 3, 'plausible': 3, 'irrelevant': 2})
            converted_rows = [json.loads(line) for line in output_samples.read_text().splitlines()]
            self.assertEqual(len(converted_rows), 8)
            self.assertEqual(converted_rows[0]['selected_action'], 'app1')
            self.assertEqual(converted_rows[0]['reward'], 1.0)
            self.assertEqual(converted_rows[3]['reward'], 0.6)
            self.assertEqual(converted_rows[-1]['reward'], 0.0)

    def test_convert_raw_sequence_to_v6_expanded_app_respects_repeat_counts(self) -> None:
        rows = [
            {
                'episode_id': 'arrive_office_ep01',
                'scenario_id': 'ARRIVE_OFFICE',
                'scenario_elapsed_sec': 0,
                'emit_recommendation': 1,
                'gt_app': 'app1',
                'features': _base_features(),
            }
        ]
        markdown = "\n".join(
            [
                "# v6",
                "",
                "## Scenario Defaults",
                "",
                "| scenarioId | scenarioNameZh | most relevant 3 R/O actionIds | other plausible 3 R/O actionIds | irrelevant 2 R/O actionIds | most relevant 3 app categories | other plausible 3 app categories | irrelevant 2 app categories |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "| `ARRIVE_OFFICE` | 到达办公室 | `A1`<br>`A2`<br>`A3` | `A4`<br>`A5`<br>`A6` | `A7`<br>`A8` | `app1`<br>`app2`<br>`app3` | `app4`<br>`app5`<br>`app6` | `app7`<br>`app8` |",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / 'raw.jsonl'
            input_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            output_samples = tmp_path / 'expanded_app_repeat.jsonl'
            relevance_path = tmp_path / 'v6.md'
            relevance_path.write_text(markdown)

            summary = convert_raw_sequence_to_v6_expanded_app(
                input_path=input_path,
                output_samples_path=output_samples,
                relevance_markdown=relevance_path,
                most_relevant_repeat=2,
                plausible_repeat=1,
                irrelevant_repeat=1,
            )

            self.assertEqual(summary.emitted_samples, 11)
            self.assertEqual(summary.tier_counts, {'most_relevant': 6, 'plausible': 3, 'irrelevant': 2})
            converted_rows = [json.loads(line) for line in output_samples.read_text().splitlines()]
            self.assertEqual(len(converted_rows), 11)
            self.assertTrue(converted_rows[0]['event_id'].endswith(':r1'))

    def test_convert_raw_sequence_to_v6_expanded_app_allows_zero_plausible_and_irrelevant_repeat(self) -> None:
        rows = [
            {
                'episode_id': 'arrive_office_ep01',
                'scenario_id': 'ARRIVE_OFFICE',
                'scenario_elapsed_sec': 0,
                'emit_recommendation': 1,
                'gt_app': 'app1',
                'features': _base_features(),
            }
        ]
        markdown = "\n".join(
            [
                "# v6",
                "",
                "## Scenario Defaults",
                "",
                "| scenarioId | scenarioNameZh | most relevant 3 R/O actionIds | other plausible 3 R/O actionIds | irrelevant 2 R/O actionIds | most relevant 3 app categories | other plausible 3 app categories | irrelevant 2 app categories |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "| `ARRIVE_OFFICE` | 到达办公室 | `A1`<br>`A2`<br>`A3` | `A4`<br>`A5`<br>`A6` | `A7`<br>`A8` | `app1`<br>`app2`<br>`app3` | `app4`<br>`app5`<br>`app6` | `app7`<br>`app8` |",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / 'raw.jsonl'
            input_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            output_samples = tmp_path / 'expanded_app_only_relevant.jsonl'
            relevance_path = tmp_path / 'v6.md'
            relevance_path.write_text(markdown)

            summary = convert_raw_sequence_to_v6_expanded_app(
                input_path=input_path,
                output_samples_path=output_samples,
                relevance_markdown=relevance_path,
                most_relevant_repeat=1,
                plausible_repeat=0,
                irrelevant_repeat=0,
            )

            self.assertEqual(summary.emitted_samples, 3)
            self.assertEqual(summary.tier_counts, {'most_relevant': 3})
            converted_rows = [json.loads(line) for line in output_samples.read_text().splitlines()]
            self.assertEqual(len(converted_rows), 3)
            self.assertTrue(all(row['source_relevance_tier'] == 'most_relevant' for row in converted_rows))


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.raw_synthetic import convert_raw_sequence_to_v0_app


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


if __name__ == '__main__':
    unittest.main()

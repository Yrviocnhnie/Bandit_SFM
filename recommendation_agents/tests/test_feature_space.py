from __future__ import annotations

import unittest

from recommendation_agents.feature_space import V0FeatureSpace


class FeatureSpaceTest(unittest.TestCase):
    def test_v0_dimension_matches_shared_action_contract(self) -> None:
        feature_space = V0FeatureSpace()
        self.assertEqual(feature_space.dimension, 306)
        names = feature_space.feature_names()
        self.assertIn('precondition=unknown', names)
        self.assertFalse(any(name.startswith('scenarioId=') for name in names))

    def test_unknown_categories_and_previous_state_aliases_encode_cleanly(self) -> None:
        feature_space = V0FeatureSpace()
        vector = feature_space.encode(
            {
                'state_current': 'missing_state_code',
                'state_previous': 'office_arriving',
                'state_duration_sec': 120,
                'ps_time': 'morning',
                'hour': 9,
                'cal_hasUpcoming': 0,
                'ps_dayType': 'workday',
                'ps_motion': 'mystery_motion',
                'wifiLost': 0,
                'wifiLostCategory': 'mystery_place',
                'cal_eventCount': 0,
                'cal_inMeeting': 0,
                'cal_nextLocation': 'mystery_place',
                'ps_sound': 'quiet',
                'sms_delivery_pending': 0,
                'sms_train_pending': 0,
                'sms_flight_pending': 0,
                'sms_hotel_pending': 0,
                'sms_movie_pending': 0,
                'sms_hospital_pending': 0,
                'sms_ride_pending': 0,
                'timestep': 1,
                'ps_location': 'mystery_place',
                'ps_phone': 'unknown',
                'batteryLevel': 50,
                'isCharging': 0,
                'networkType': 'wifi',
                'activityState': 'mystery_activity',
                'activityDuration': 10,
                'user_id_hash_bucket': 'b07',
                'age_bucket': 'unknown',
                'sex': 'unknown',
                'has_kids': 0,
            }
        )
        self.assertEqual(vector.shape, (306,))
        self.assertGreater(vector.sum(), 0.0)


if __name__ == '__main__':
    unittest.main()

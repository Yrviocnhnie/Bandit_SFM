from __future__ import annotations

import unittest

from recommendation_agents.feature_space import V0FeatureSpace


class FeatureSpaceTest(unittest.TestCase):
    def test_v0_dimension_matches_report(self) -> None:
        feature_space = V0FeatureSpace()
        self.assertEqual(feature_space.dimension, 314)

    def test_unknown_categories_fallback_cleanly(self) -> None:
        feature_space = V0FeatureSpace()
        vector = feature_space.encode(
            "ARRIVE_OFFICE",
            {
                "state_current": "missing_state_code",
                "ps_time": "morning",
                "hour": 9,
                "cal_hasUpcoming": 0,
                "ps_dayType": "workday",
                "ps_motion": "mystery_motion",
                "wifiLost": 0,
                "wifiLostCategory": "mystery_place",
                "cal_eventCount": 0,
                "cal_inMeeting": 0,
                "cal_nextLocation": "mystery_place",
                "ps_sound": "quiet",
                "sms_delivery_pending": 0,
                "sms_train_pending": 0,
                "sms_flight_pending": 0,
                "sms_hotel_pending": 0,
                "sms_movie_pending": 0,
                "sms_hospital_pending": 0,
                "sms_ride_pending": 0,
                "timestep": 1,
                "ps_location": "mystery_place",
                "ps_phone": "unknown",
                "batteryLevel": 50,
                "isCharging": 0,
                "networkType": "wifi",
                "transportMode": "mystery_transport",
                "activityState": "mystery_activity",
                "activityDuration": 10,
                "user_id_hash_bucket": "b0",
                "age_bucket": "unknown",
                "sex": "unknown",
                "has_kids": 0,
            },
        )
        self.assertEqual(vector.shape, (314,))
        self.assertGreater(vector.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()


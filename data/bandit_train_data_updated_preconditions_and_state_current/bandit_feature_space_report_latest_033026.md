# Final Feature Space Report for the Bandit Agent

## 1. Scope and versioning

This is the final cleaned feature-space specification after the latest update.

- **v0 = initial offline training**: predict the current default target recommendation from independent scenario snapshots.
- **v1 = online user-feedback learning**: adapt recommendations using recent accepted actions, short history, and profile context.

The same shared feature taxonomy is used for both the **RO model** and the **App model**.

- In **v0**, long history features are **off**.
- In **v1**, history and personalization features are **on**.

## 2. Final design decisions applied in this report

1. **`scenarioId` is removed from the model input vector.**
   - It is still used at serving time for **scenario routing / candidate masking**, so the model only ranks actions valid for the matched scenario.
2. **`precondition` is kept in the input vector.**
   - This is a compact summarized prior-state feature, not the full trajectory history.
   - It is allowed in **v0** because it can be provided as a single summarized field even when whole-day trajectories are not available.
3. If a taxonomy already includes `unknown`, I do **not** add another extra `unknown`.
4. **Binary features stay strictly binary**: `{0,1}` only.
5. `cal_nextMinutes` is **removed** from the final model input.
6. In the **StateCode appendix**, I keep only the **64 canonical StateCode values**. I do **not** append an extra generic `unknown` there.
7. Raw user name text is not used directly. It is represented by `user_id_hash_bucket`.
8. The output action space includes a global **`NONE`** fallback arm.

## 3. Verified files and current coverage

- `default_rules_1_english.json`: **65** scenarios / rules
- `scenario_recommendation_actions_v4_english.md`: **65** scenarios
- Total `R&O` candidate actions in v4: **167**
- Default RO targets in v4: **65**
- App-category taxonomy size used by v4: **10**
- App categories currently observed in v4 defaults: **9**

### Important schema note

The rules use these SMS-derived fields:

- `sms_delivery_pending`
- `sms_train_pending`
- `sms_flight_pending`
- `sms_hotel_pending`
- `sms_movie_pending`
- `sms_hospital_pending`
- `sms_ride_pending`

If any of these are still missing in the runtime DataTray schema, they should be added before full end-to-end training and serving.

## 4. Feature-frequency statistics from the scenario rules

This table is about **rule usage frequency**, not the final keep/drop decision. So `cal_nextMinutes` is still shown here because it appears in the handwritten rules, even though it is excluded from the final shared input vector.

| feature | scenarios using it | coverage | total uses | in conditions | in preconditions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `state_current` | 54 | 83.1% | 62 | 52 | 10 |
| `state_duration_sec` | 11 | 16.9% | 11 | 11 | 0 |
| `ps_time` | 5 | 7.7% | 5 | 5 | 0 |
| `hour` | 4 | 6.2% | 6 | 6 | 0 |
| `cal_hasUpcoming` | 3 | 4.6% | 3 | 3 | 0 |
| `cal_nextMinutes` | 2 | 3.1% | 2 | 2 | 0 |
| `ps_dayType` | 2 | 3.1% | 2 | 2 | 0 |
| `ps_motion` | 2 | 3.1% | 2 | 2 | 0 |
| `wifiLost` | 2 | 3.1% | 2 | 2 | 0 |
| `wifiLostCategory` | 2 | 3.1% | 2 | 2 | 0 |
| `cal_eventCount` | 1 | 1.5% | 1 | 1 | 0 |
| `cal_inMeeting` | 1 | 1.5% | 1 | 1 | 0 |
| `cal_nextLocation` | 1 | 1.5% | 1 | 1 | 0 |
| `ps_sound` | 1 | 1.5% | 1 | 1 | 0 |
| `sms_delivery_pending` | 1 | 1.5% | 1 | 1 | 0 |
| `sms_flight_pending` | 1 | 1.5% | 1 | 1 | 0 |
| `sms_hospital_pending` | 1 | 1.5% | 1 | 1 | 0 |
| `sms_hotel_pending` | 1 | 1.5% | 1 | 1 | 0 |
| `sms_movie_pending` | 1 | 1.5% | 1 | 1 | 0 |
| `sms_ride_pending` | 1 | 1.5% | 1 | 1 | 0 |
| `sms_train_pending` | 1 | 1.5% | 1 | 1 | 0 |

### Interpretation

- `state_current` is still the dominant rule feature.
- `state_duration_sec`, `ps_time`, and `hour` are the main secondary disambiguators.
- Calendar, Wi-Fi transition, sound, and SMS fields are lower-frequency but still necessary for a few scenarios.
- `preconditions` appear in **10** scenarios, and all of them are `state_current`-based prior-state checks with `withinMs` windows.
- `cal_nextMinutes` is low-frequency and therefore removed from the final shared input vector.

## 5. `state_current` and `precondition` usage statistics

### 5.1 Canonical state-code coverage in the rules

- Canonical `StateCode` values in the data spec: **64**
- Canonical `StateCode` values used in the rules: **57 / 64 = 89.1%**
- Model dimension for `state_current`: **64**
- Model dimension for `precondition`: **65** = 64 canonical `StateCode` values + `none`

### 5.2 Legacy labels normalized to current spec labels

- `at_education_quiet` -> `at_education_class`
- `at_education_noisy` -> `at_education_break`

### 5.3 Canonical `StateCode` values not used by current rules

`at_custom`, `at_health`, `at_health_inpatient`, `home_active`, `in_transit`, `outdoor_resting`, `walking_unknown`

### 5.4 Most reused `state_current` values

| state_current | scenarios using it | total mentions |
| --- | ---: | ---: |
| `office_working` | 9 | 9 |
| `office_working_focused` | 7 | 7 |
| `at_cafe_quiet` | 6 | 6 |
| `home_daytime_rest` | 6 | 6 |
| `home_morning_workday` | 6 | 6 |
| `office_overtime` | 6 | 6 |
| `at_cafe` | 5 | 5 |
| `at_gym` | 5 | 5 |
| `at_gym_exercising` | 5 | 5 |
| `home_daytime_workday` | 5 | 5 |
| `home_evening` | 5 | 5 |
| `home_morning_rest` | 5 | 5 |
| `office_working_noisy` | 5 | 5 |
| `office_arriving` | 4 | 4 |
| `outdoor_walking` | 4 | 4 |

### 5.5 Scenarios with explicit preconditions

`OFFICE_LUNCH_OUT`, `LEAVE_OFFICE`, `LATE_RETURN_HOME`, `COMMUTE_MORNING`, `COMMUTE_EVENING`, `HOME_TO_GYM`, `RETURN_OFFICE_AFTER_LUNCH`, `HOME_AFTER_GYM`, `COMMUTE_FROM_HOME`, `OFFICE_TO_CAFE`

These are all prior-`state_current` constraints with `withinMs`, so representing them as a compact categorical `precondition` feature is reasonable.

## 6. Final feature space for the bandit agent

The final model input features are grouped into:

1. **Occurs in rules**
2. **Auxiliary features**
3. **History features**
4. **User profile**

### 6.1 Final included features table

| group | feature | v0 | v1 | feature type | feature classes | encoding | dim | rationale |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| Occurs in rules | `state_current` | Yes | Yes | categorical | 64 canonical `StateCode` values | one-hot | 64 | Main semantic summary of the current situation. No extra generic `unknown` is added because the state-code space already includes several explicit unknown-like states. |
| Occurs in rules | `precondition` | Yes | Yes | categorical | 64 canonical `StateCode` values + `none` | one-hot | 65 | Compact summary of the matched rule precondition. `none` is used when the scenario has no explicit precondition. |
| Occurs in rules | `state_duration_sec` | Yes | Yes | numeric | seconds >= 0 | 1 normalized scalar | 1 | Distinguishes short vs long dwell/session cases. |
| Occurs in rules | `ps_time` | Yes | Yes | categorical | `sleeping, dawn, morning, forenoon, lunch, afternoon, evening, night, late_night` | one-hot | 9 | Coarse time-of-day abstraction used directly by several rules. |
| Occurs in rules | `hour` | Yes | Yes | categorical | `0..23` | one-hot | 24 | Exact hour bucket used by timing-specific rules. |
| Occurs in rules | `cal_hasUpcoming` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Calendar-driven scenario signal. |
| Occurs in rules | `ps_dayType` | Yes | Yes | categorical | `workday, weekend, holiday` | one-hot | 3 | Workday vs weekend vs holiday split. |
| Occurs in rules | `ps_motion` | Yes | Yes | categorical | `stationary, walking, running, cycling, driving, transit, unknown` | one-hot | 7 | Motion pattern used in a few rules and useful more broadly. |
| Occurs in rules | `wifiLost` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Strong transition trigger for leaving-location scenarios. |
| Occurs in rules | `wifiLostCategory` | Yes | Yes | categorical | 17 `LocationCategory` classes | one-hot | 17 | Which category of place was just left. |
| Occurs in rules | `cal_eventCount` | Yes | Yes | numeric | count >= 0 | 1 normalized scalar | 1 | Future-event density signal. |
| Occurs in rules | `cal_inMeeting` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Current meeting flag. |
| Occurs in rules | `cal_nextLocation` | Yes | Yes | categorical | 17 `LocationCategory` classes | one-hot | 17 | Map next event location into the same location taxonomy. |
| Occurs in rules | `ps_sound` | Yes | Yes | categorical | `silent, quiet, normal, noisy, unknown` | one-hot | 5 | Needed for noisy/quiet scenarios; useful for social and meeting contexts. |
| Occurs in rules | `sms_delivery_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Specific parcel/delivery scenario signal. |
| Occurs in rules | `sms_train_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Specific travel-booking scenario signal. |
| Occurs in rules | `sms_flight_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Specific travel-booking scenario signal. |
| Occurs in rules | `sms_hotel_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Specific travel-booking scenario signal. |
| Occurs in rules | `sms_movie_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Specific event-booking scenario signal. |
| Occurs in rules | `sms_hospital_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Specific appointment scenario signal. |
| Occurs in rules | `sms_ride_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Specific rideshare scenario signal. |
| Auxiliary | `timestep` | Yes | Yes | numeric | integer in `[0, 86400]` | 1 normalized scalar | 1 | Fine-grained second-of-day signal. |
| Auxiliary | `ps_location` | Yes | Yes | categorical | 17 `LocationCategory` classes | one-hot | 17 | Useful generalization signal beyond exact state codes. |
| Auxiliary | `ps_phone` | Yes | Yes | categorical | `in_use, holding_lying, on_desk, face_up, in_pocket, face_down, charging, unknown` | one-hot | 8 | Useful for substate refinement and user-attention context. |
| Auxiliary | `batteryLevel` | Yes | Yes | numeric | `0..100` | 1 normalized scalar | 1 | Operational context for whether certain actions/apps are suitable. |
| Auxiliary | `isCharging` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Useful for battery-sensitive recommendations. |
| Auxiliary | `networkType` | Yes | Yes | categorical | `wifi, cellular, none` | one-hot | 3 | Connectivity-sensitive app/operation feasibility. |
| Auxiliary | `transportMode` | Yes | Yes | categorical | `walking, running, cycling, driving, transit, stationary, unknown` | one-hot | 7 | Stable mobility abstraction for generalization. `unknown` is retained at model level because the runtime signal can be missing even though the base spec omits it. |
| Auxiliary | `activityState` | Yes | Yes | categorical | `sitting, sleeping, standing, active, unknown` | one-hot | 5 | High-level activity summary. `unknown` is retained at model level for missing or weak sensor cases. |
| Auxiliary | `activityDuration` | Yes | Yes | numeric | seconds >= 0 | 1 normalized scalar | 1 | Persistence of current activity state. |
| History (v1 only) | `hist_state_t-1` | No | Yes | categorical | 64 canonical `StateCode` values + `none` | one-hot | 65 | Most recent previous state in the short trajectory. |
| History (v1 only) | `hist_state_t-2` | No | Yes | categorical | 64 canonical `StateCode` values + `none` | one-hot | 65 | Second previous state in the short trajectory. |
| History (v1 only) | `hist_state_t-3` | No | Yes | categorical | 64 canonical `StateCode` values + `none` | one-hot | 65 | Third previous state in the short trajectory. |
| History (v1 only) | `prev_location` | No | Yes | categorical | 17 `LocationCategory` classes | one-hot | 17 | Previous location category; missing can be folded into existing `unknown`. |
| History (v1 only) | `prev_activityState` | No | Yes | categorical | `sitting, sleeping, standing, active, unknown` | one-hot | 5 | Previous activity summary. |
| History (v1 only) | `recent_accepted_ro_action_in_scenario` | No | Yes | categorical | 167 scenario-scoped R/O actions + `none` | one-hot | 168 | Core personalization signal for recent accepted R/O behavior in the same scenario. |
| History (v1 only) | `recent_accepted_app_category_in_scenario` | No | Yes | categorical | 10 app categories + `none` | one-hot | 11 | Core personalization signal for recent accepted app-category choice in the same scenario. |
| History (v1 only) | `recent_ro_feedback_score_same_scenario` | No | Yes | numeric | `[-1,1]` | 1 scalar | 1 | Aggregated recent reward for R/O choices in the same scenario. |
| History (v1 only) | `recent_app_feedback_score_same_scenario` | No | Yes | numeric | `[-1,1]` | 1 scalar | 1 | Aggregated recent reward for app-category choices in the same scenario. |
| User profile | `user_id_hash_bucket` | Yes | Yes | categorical | 32 buckets: `b0..b31` | one-hot | 32 | Stable anonymized user identity proxy instead of raw name. |
| User profile | `age_bucket` | Yes | Yes | categorical | `0_17, 18_24, 25_34, 35_44, 45_54, 55_64, 65_plus, unknown` | one-hot | 8 | Safer and more stable than raw age. |
| User profile | `sex` | Yes | Yes | categorical | `female, male, other, unknown` | one-hot | 4 | Optional demographic feature; include only if allowed and consented. |
| User profile | `has_kids` | Yes | Yes | binary | `{0,1}` | binary int | 1 | Family-routine signal. |

### 6.2 Features intentionally excluded from the final shared vector

| feature | status | reason |
| --- | --- | --- |
| `scenarioId` | Excluded from input | Removed from the input vector per your latest update. It is still used outside the model for scenario routing and candidate masking. |
| `cal_nextMinutes` | Excluded | Low frequency in rules and not very useful compared with `cal_hasUpcoming`, `cal_eventCount`, `cal_inMeeting`, and `cal_nextLocation`. |
| `ps_light` | Excluded | Light effects are already captured reasonably well by the state encoder and the dark substates. |
| `speed` | Excluded | Likely noisy and largely redundant with `ps_motion` and `transportMode`. |
| raw `wifiSsid`, raw `geofence`, raw lat/lon, raw `cellId` | Excluded | Too sparse and device/environment specific for the first clean UCB feature space. |
| raw calendar titles | Excluded | Too sparse and privacy-sensitive for the first version. |
| raw user name text | Excluded | Replaced by `user_id_hash_bucket`. |

### 6.3 Why this final feature space makes sense

- **For v0**: the model sees all the rule-used features that matter, a small set of useful auxiliary signals, and user profile. That is enough to learn the current default policy from independent scenario snapshots.
- **For v0**, `precondition` is still allowed because it is a **single summarized prerequisite-state field**, not a full history tensor.
- **For v1**: the model adds short trajectory history and recent accepted-action memory, so it can personalize within the same scenario.
- **Binary features remain binary**, which keeps the representation cleaner and avoids artificial tri-state expansion.
- **StateCode stays canonical (64)** in the appendix and in the main `state_current` feature, which is more consistent with the actual encoder design.

## 7. Dimension summary

### 7.1 Input dimension by feature group

| group | v0 dim | v1 dim |
| --- | ---: | ---: |
| Occurs in rules | 223 | 223 |
| Auxiliary | 44 | 44 |
| History (v1 only) | 0 | 398 |
| User profile | 45 | 45 |
| **Total** | **312** | **710** |

### 7.2 Final shared input shapes

- **v0 input shape**: `(312,)`
- **v1 input shape**: `(710,)`
- Batch shapes are `(batch_size, 312)` and `(batch_size, 710)`

## 8. Output space

| model | phase | output space | output dim | notes |
| --- | --- | --- | ---: | --- |
| RO model | v0 supervised target | default RO target + `NONE` | 66 | 65 default scenario targets + 1 global `NONE` fallback. |
| RO model | v1 online arm pool | scenario-valid R/O candidate actions + `NONE` | 168 | 167 R/O actions in v4 + 1 global `NONE` fallback. |
| App model | v0/v1 model head | app-category taxonomy + `NONE` | 11 | 10 app categories + 1 global `NONE` fallback. |

### Practical interpretation

- **RO model**: use the shared context to learn the default target in v0, then rank the full scenario-valid R/O arm pool in v1.
- **App model**: predict **app category**, not a concrete app. A separate mapping layer can convert category -> concrete app for the user.
- **`scenarioId` is not part of the input vector anymore**, but it should still be used as a serving-time mask so the agent only ranks actions valid for that scenario.

## 9. Final recommendation

The final design is:

- **v0**: offline model with rule-used features + auxiliary features + user profile, and with **`precondition` included** but **history features off**
- **v1**: same base context plus past-3-state trajectory, previous location/activity, recent accepted actions, and feedback aggregates
- **`scenarioId` removed from the input vector**
- **`scenarioId` retained as serving-time routing / candidate masking metadata**
- **StateCode appendix kept canonical at 64 values with no extra generic `unknown` appended**

This is a realistic feature space for a bandit/UCB agent because it starts with the same signals the rules already use, stays compact for offline training, and then cleanly expands to online personalization once feedback becomes available.

## Appendix A. Scenario IDs (65 canonical, serving-time routing only)

`ARRIVE_OFFICE`, `OFFICE_LUNCH_OUT`, `OFFICE_AFTERNOON`, `LEAVE_OFFICE`, `WEEKDAY_HOME_DAY`, `ARRIVE_TRANSIT_HUB`, `LATE_NIGHT_OVERTIME`, `WEEKEND_OVERTIME`, `EVENING_AT_OFFICE`, `MORNING_EXERCISE`, `SHOPPING`, `HOME_EVENING`, `ARRIVE_DINING`, `HOME_LATE_NIGHT`, `WEEKEND_MORNING`, `WEEKEND_OUTING`, `LATE_RETURN_HOME`, `HOME_AFTERNOON`, `MORNING_COFFEE`, `GYM_WORKOUT`, `GYM_REST`, `GYM_LONG_STAY`, `OUTDOOR_RUNNING`, `WEEKEND_OUTDOOR_WALK`, `CYCLING`, `COMMUTE_MORNING`, `COMMUTE_EVENING`, `HOME_TO_GYM`, `WAKE_UP`, `PARK_WALK`, `LONG_DRIVE`, `CAFE_STAY`, `QUIET_HOME`, `NOISY_GATHERING`, `LONG_OUTDOOR_WALK`, `LATE_NIGHT_NOISY`, `DELIVERY_AT_OFFICE`, `EDUCATION_WAITING`, `EDUCATION_LONG_SIT`, `HOME_LUNCH`, `AFTER_LUNCH_WALK`, `MEETING_UPCOMING`, `IN_MEETING`, `REMOTE_MEETING`, `NO_MEETINGS`, `CAL_HAS_EVENTS`, `RETURN_OFFICE_AFTER_LUNCH`, `HOME_AFTER_GYM`, `COMMUTE_FROM_HOME`, `OFFICE_TO_CAFE`, `LATE_NIGHT_PHONE`, `OFFICE_FOCUS_LONG`, `HOME_DARK_LONG`, `UNKNOWN_LONG_STAY`, `OFFICE_LONG_SESSION`, `HOME_EVENING_DARK`, `HOME_EVENING_NOISY`, `SCHOOL_QUIET`, `CAFE_QUIET`, `TRAIN_DEPARTURE`, `FLIGHT_BOARDING`, `HOTEL_CHECKIN`, `MOVIE_TICKET`, `HOSPITAL_APPOINTMENT`, `RIDESHARE_PICKUP`

## Appendix B. StateCode values (64 canonical)

`home_sleeping`, `home_morning_workday`, `home_morning_rest`, `home_daytime_workday`, `home_daytime_rest`, `home_evening`, `home_active`, `office_arriving`, `office_lunch_break`, `office_working`, `office_overtime`, `office_late_overtime`, `office_rest_day`, `commuting_walk_out`, `commuting_walk_home`, `commuting_cycle_out`, `commuting_cycle_home`, `commuting_drive_out`, `commuting_drive_home`, `commuting_transit_out`, `commuting_transit_home`, `driving`, `in_transit`, `at_metro`, `at_rail_station`, `at_airport`, `at_transit_hub`, `outdoor_walking`, `outdoor_running`, `outdoor_cycling`, `outdoor_resting`, `at_restaurant_lunch`, `at_restaurant_dinner`, `at_restaurant_other`, `at_cafe`, `at_gym_exercising`, `at_gym`, `at_shopping`, `at_health`, `at_social`, `at_education`, `at_custom`, `stationary_unknown`, `walking_unknown`, `home_sleeping_lying`, `home_morning_workday_lying`, `home_morning_rest_lying`, `home_daytime_workday_dark`, `home_daytime_workday_lying`, `home_daytime_rest_dark`, `home_daytime_rest_lying`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy`, `office_working_focused`, `office_working_noisy`, `at_cafe_quiet`, `at_education_class`, `at_education_break`, `at_health_inpatient`, `unknown_noisy`, `unknown_dark`, `unknown_settled`, `unknown_lying`

## Appendix C. LocationCategory values (17)

`home`, `work`, `restaurant`, `cafe`, `gym`, `metro`, `rail_station`, `airport`, `transit`, `shopping`, `outdoor`, `health`, `social`, `education`, `custom`, `en_route`, `unknown`

## Appendix D. Other categorical taxonomies used directly

- `TimeSlot` (9): `sleeping`, `dawn`, `morning`, `forenoon`, `lunch`, `afternoon`, `evening`, `night`, `late_night`
- `MotionCategory` (7): `stationary`, `walking`, `running`, `cycling`, `driving`, `transit`, `unknown`
- `PhoneCategory` (8): `in_use`, `holding_lying`, `on_desk`, `face_up`, `in_pocket`, `face_down`, `charging`, `unknown`
- `SoundCategory` (5): `silent`, `quiet`, `normal`, `noisy`, `unknown`
- `DayType` (3): `workday`, `weekend`, `holiday`
- `NetworkType` (3): `wifi`, `cellular`, `none`
- `TransportMode` (7 model classes): `walking`, `running`, `cycling`, `driving`, `transit`, `stationary`, `unknown`
- `ActivityState` (5 model classes): `sitting`, `sleeping`, `standing`, `active`, `unknown`
- `AgeBucket` (8): `0_17`, `18_24`, `25_34`, `35_44`, `45_54`, `55_64`, `65_plus`, `unknown`
- `Sex` (4): `female`, `male`, `other`, `unknown`
- App taxonomy (10): `social`, `productivity`, `entertainment`, `navigation`, `shopping`, `news`, `health`, `music`, `reading`, `game`

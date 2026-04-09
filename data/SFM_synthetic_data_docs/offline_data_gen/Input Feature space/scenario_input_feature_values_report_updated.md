# Scenario Input Feature Values Report (State-Code Aligned)

## 1. Scope

This updated report lists the **possible v0 input feature values for every scenario**, but now explicitly enforces **state-code alignment**.

The guiding rule is:

- choose a valid `state_current` for the scenario,
- then assign all **state_code dependent** and **state_code aligned** features so they remain consistent with that chosen state,
- and only then assign the remaining rule-dependent or free-context features.

This means the report is no longer treating all features as independent once a scenario is chosen.

## 2. Dependency legend

- **state_code dependent**: directly constrained by the chosen `state_current`
- **state_code dependent (substate-sensitive)**: forced only when the chosen state is a refined substate
- **state_code aligned**: not directly encoded by `state_current`, but should remain realistic given the chosen state
- **rule dependent**: driven by explicit rule conditions or scenario logic
- **profile**: independently sampled profile inputs

## 3. Scenario-wise possible values

### ARRIVE_OFFICE — Arrive office

- **Category:** `work`
- **Allowed `state_current` values:** `office_arriving`, `office_working`, `office_lunch_break`, `office_overtime`, `office_late_overtime`, `office_rest_day`
- **Allowed `precondition` values:** `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out`
- **Unique relevant categorical samples:** **25,920** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_arriving`, `office_working`, `office_lunch_break`, `office_overtime`, `office_late_overtime`, `office_rest_day` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### OFFICE_LUNCH_OUT — Office lunch out

- **Category:** `work`
- **Allowed `state_current` values:** `outdoor_walking`, `at_restaurant_lunch`, `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition` values:** `office_arriving`, `office_working`, `office_lunch_break`
- **Unique relevant categorical samples:** **1,752** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `outdoor_walking`, `at_restaurant_lunch`, `at_cafe`, `at_cafe_quiet` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_arriving`, `office_working`, `office_lunch_break` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `forenoon`, `lunch` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `[11, 13]` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `1` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `work` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `cafe`, `outdoor`, `restaurant` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### OFFICE_AFTERNOON — Office afternoon

- **Category:** `work`
- **Allowed `state_current` values:** `office_working`, `office_working_focused`, `office_working_noisy`
- **Allowed `precondition` values:** `office_arriving`, `office_working`, `office_lunch_break`
- **Unique relevant categorical samples:** **2,208** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_working`, `office_working_focused`, `office_working_noisy` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_arriving`, `office_working`, `office_lunch_break` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### LEAVE_OFFICE — Leave office

- **Category:** `commute`
- **Allowed `state_current` values:** `commuting_walk_home`, `commuting_cycle_home`, `commuting_drive_home`, `commuting_transit_home`, `outdoor_walking`
- **Allowed `precondition` values:** `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy`
- **Unique relevant categorical samples:** **648** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `commuting_walk_home`, `commuting_cycle_home`, `commuting_drive_home`, `commuting_transit_home`, `outdoor_walking` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `evening`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `[17, 20]` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `transit`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `1` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `work` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `en_route`, `outdoor` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `transit`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### WEEKDAY_HOME_DAY — Weekday home daytime

- **Category:** `home`
- **Allowed `state_current` values:** `home_daytime_workday`, `home_daytime_workday_dark`, `home_daytime_workday_lying`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_workday_lying`
- **Unique relevant categorical samples:** **960** (from `scenario_unique_samples_count.md`)
- **Substate note:** if a selected state requires light refinement, valid trigger values include `dark`.

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_daytime_workday`, `home_daytime_workday_dark`, `home_daytime_workday_lying` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_morning_workday`, `home_morning_workday_lying` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon`, `lunch` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dark`, `dim`, `normal`, `bright` | 4 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### ARRIVE_TRANSIT_HUB — Arrive transit hub

- **Category:** `travel`
- **Allowed `state_current` values:** `at_metro`, `at_rail_station`, `at_airport`, `at_transit_hub`
- **Allowed `precondition` values:** `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out`
- **Unique relevant categorical samples:** **6,912** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_metro`, `at_rail_station`, `at_airport`, `at_transit_hub` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `forenoon`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `airport`, `metro`, `rail_station`, `transit` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### LATE_NIGHT_OVERTIME — Late night overtime

- **Category:** `work`
- **Allowed `state_current` values:** `office_overtime`, `office_late_overtime`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `office_working_noisy`, `office_overtime`
- **Unique relevant categorical samples:** **3,840** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_overtime`, `office_late_overtime` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_working`, `office_working_focused`, `office_working_noisy`, `office_overtime` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `evening`, `late_night`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### WEEKEND_OVERTIME — Weekend overtime

- **Category:** `work`
- **Allowed `state_current` values:** `office_rest_day`
- **Allowed `precondition` values:** `office_arriving`, `office_rest_day`
- **Unique relevant categorical samples:** **3,840** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_rest_day` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_arriving`, `office_rest_day` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon`, `lunch`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### EVENING_AT_OFFICE — Evening at office

- **Category:** `work`
- **Allowed `state_current` values:** `office_overtime`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `office_working_noisy`, `office_overtime`
- **Unique relevant categorical samples:** **1,920** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_overtime` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_working`, `office_working_focused`, `office_working_noisy`, `office_overtime` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `evening`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### WAKE_UP — Wake up

- **Category:** `morning`
- **Allowed `state_current` values:** `home_morning_workday`, `home_morning_rest`
- **Allowed `precondition` values:** `home_sleeping`, `home_sleeping_lying`
- **Unique relevant categorical samples:** **1,152** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_morning_workday`, `home_morning_rest` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_sleeping`, `home_sleeping_lying` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `dawn`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### MORNING_EXERCISE — Morning exercise

- **Category:** `exercise`
- **Allowed `state_current` values:** `at_gym`, `at_gym_exercising`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_rest`, `home_morning_rest_lying`
- **Unique relevant categorical samples:** **1,944** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_gym`, `at_gym_exercising` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_morning_workday`, `home_morning_rest`, `home_morning_rest_lying` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `running`, `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `gym` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `running`, `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### SHOPPING — Shopping

- **Category:** `shopping`
- **Allowed `state_current` values:** `at_shopping`
- **Allowed `precondition` values:** `commuting_drive_out`, `commuting_walk_out`, `outdoor_walking`
- **Unique relevant categorical samples:** **2,592** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_shopping` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_drive_out`, `commuting_walk_out`, `outdoor_walking` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `shopping` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### HOME_EVENING — Home evening

- **Category:** `home`
- **Allowed `state_current` values:** `home_evening`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy`
- **Allowed `precondition` values:** `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home`
- **Unique relevant categorical samples:** **8,832** (from `scenario_unique_samples_count.md`)
- **Substate note:** if a selected state requires light refinement, valid trigger values include `dark`.

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_evening`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `evening`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dark`, `dim`, `normal` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### ARRIVE_DINING — Arrive restaurant/cafe

- **Category:** `dining`
- **Allowed `state_current` values:** `at_restaurant_lunch`, `at_restaurant_dinner`, `at_restaurant_other`, `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition` values:** `outdoor_walking`, `at_shopping`, `commuting_walk_out`, `commuting_drive_out`
- **Unique relevant categorical samples:** **16,128** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_restaurant_lunch`, `at_restaurant_dinner`, `at_restaurant_other`, `at_cafe`, `at_cafe_quiet` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `outdoor_walking`, `at_shopping`, `commuting_walk_out`, `commuting_drive_out` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `lunch`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `cafe`, `restaurant` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal` | 2 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### HOME_LATE_NIGHT — Home late night

- **Category:** `sleep`
- **Allowed `state_current` values:** `home_sleeping`, `home_sleeping_lying`
- **Allowed `precondition` values:** `home_evening`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy`
- **Unique relevant categorical samples:** **3,072** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_sleeping`, `home_sleeping_lying` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_evening`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `late_night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dark`, `dim`, `normal` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `sitting`, `sleeping` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### WEEKEND_MORNING — Weekend morning

- **Category:** `morning`
- **Allowed `state_current` values:** `home_morning_rest`, `home_morning_rest_lying`
- **Allowed `precondition` values:** `home_sleeping`, `home_sleeping_lying`
- **Unique relevant categorical samples:** **896** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_morning_rest`, `home_morning_rest_lying` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_sleeping`, `home_sleeping_lying` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `dawn`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### WEEKEND_OUTING — Weekend outing

- **Category:** `shopping`
- **Allowed `state_current` values:** `outdoor_walking`, `at_shopping`
- **Allowed `precondition` values:** `home_morning_rest`, `home_morning_rest_lying`, `home_daytime_rest`
- **Unique relevant categorical samples:** **4,860** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `outdoor_walking`, `at_shopping` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_morning_rest`, `home_morning_rest_lying`, `home_daytime_rest` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `outdoor`, `shopping` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### PARK_WALK — Park walk

- **Category:** `exercise`
- **Allowed `state_current` values:** `outdoor_walking`
- **Allowed `precondition` values:** `home_daytime_rest`, `home_daytime_workday`, `outdoor_walking`
- **Unique relevant categorical samples:** **486** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `outdoor_walking` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_daytime_rest`, `home_daytime_workday`, `outdoor_walking` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `outdoor` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### LONG_DRIVE — Long drive

- **Category:** `commute`
- **Allowed `state_current` values:** `driving`, `commuting_drive_out`, `commuting_drive_home`
- **Allowed `precondition` values:** `commuting_drive_out`, `driving`
- **Unique relevant categorical samples:** **288** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `driving`, `commuting_drive_out`, `commuting_drive_home` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_drive_out`, `driving` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `dawn`, `evening`, `morning`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `driving` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `en_route`, `home`, `outdoor`, `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_down`, `face_up`, `holding_lying`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `driving` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `sitting` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### LATE_RETURN_HOME — Late return home

- **Category:** `home`
- **Allowed `state_current` values:** `home_sleeping`, `home_sleeping_lying`
- **Allowed `precondition` values:** `office_overtime`, `office_late_overtime`, `driving`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`
- **Unique relevant categorical samples:** **4,608** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_sleeping`, `home_sleeping_lying` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_overtime`, `office_late_overtime`, `driving`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `late_night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dark`, `dim`, `normal` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `sitting`, `sleeping` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### HOME_AFTERNOON — Home afternoon

- **Category:** `home`
- **Allowed `state_current` values:** `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying`
- **Allowed `precondition` values:** `home_morning_rest`, `home_morning_workday`, `home_daytime_workday`
- **Unique relevant categorical samples:** **5,760** (from `scenario_unique_samples_count.md`)
- **Substate note:** if a selected state requires light refinement, valid trigger values include `dark`.

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_morning_rest`, `home_morning_workday`, `home_daytime_workday` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon`, `lunch` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dark`, `dim`, `normal`, `bright` | 4 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### MORNING_COFFEE — Morning coffee

- **Category:** `dining`
- **Allowed `state_current` values:** `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_rest`, `commuting_walk_out`
- **Unique relevant categorical samples:** **6,912** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_cafe`, `at_cafe_quiet` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_morning_workday`, `home_morning_rest`, `commuting_walk_out` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `lunch`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `cafe` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal` | 2 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### CAFE_STAY — Cafe stay

- **Category:** `dining`
- **Allowed `state_current` values:** `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition` values:** `at_cafe`, `at_cafe_quiet`
- **Unique relevant categorical samples:** **4,608** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_cafe`, `at_cafe_quiet` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `at_cafe`, `at_cafe_quiet` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `lunch`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `cafe` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal` | 2 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### QUIET_HOME — Quiet home

- **Category:** `home`
- **Allowed `state_current` values:** `home_evening`, `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`
- **Allowed `precondition` values:** `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`, `home_evening`
- **Unique relevant categorical samples:** **15,840** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_evening`, `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`, `home_evening` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `lunch`, `morning`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### GYM_WORKOUT — Gym workout

- **Category:** `exercise`
- **Allowed `state_current` values:** `at_gym_exercising`
- **Allowed `precondition` values:** `at_gym`
- **Unique relevant categorical samples:** **216** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_gym_exercising` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `at_gym` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `running` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `gym` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `running` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### GYM_REST — Gym rest

- **Category:** `exercise`
- **Allowed `state_current` values:** `at_gym`
- **Allowed `precondition` values:** `at_gym_exercising`
- **Unique relevant categorical samples:** **432** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_gym` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `at_gym_exercising` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `gym` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### GYM_LONG_STAY — Gym long stay

- **Category:** `exercise`
- **Allowed `state_current` values:** `at_gym`, `at_gym_exercising`
- **Allowed `precondition` values:** `at_gym`, `at_gym_exercising`
- **Unique relevant categorical samples:** **1,296** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_gym`, `at_gym_exercising` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `at_gym`, `at_gym_exercising` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `running`, `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `gym` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `running`, `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### OUTDOOR_RUNNING — Outdoor running

- **Category:** `exercise`
- **Allowed `state_current` values:** `outdoor_running`
- **Allowed `precondition` values:** `home_morning_rest`, `home_morning_workday`, `outdoor_walking`
- **Unique relevant categorical samples:** **486** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `outdoor_running` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_morning_rest`, `home_morning_workday`, `outdoor_walking` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `running` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `unknown` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `outdoor`, `unknown` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_pocket`, `in_use`, `unknown` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `running` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### NOISY_GATHERING — Noisy gathering

- **Category:** `social`
- **Allowed `state_current` values:** `unknown_noisy`, `at_social`
- **Allowed `precondition` values:** `home_evening`, `at_restaurant_dinner`, `at_social`
- **Unique relevant categorical samples:** **4,212** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `unknown_noisy`, `at_social` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_evening`, `at_restaurant_dinner`, `at_social` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `unknown` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `social`, `unknown` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_pocket`, `in_use`, `unknown` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal` | 2 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### WEEKEND_OUTDOOR_WALK — Weekend outdoor walk

- **Category:** `exercise`
- **Allowed `state_current` values:** `outdoor_walking`
- **Allowed `precondition` values:** `home_morning_rest`, `home_daytime_rest`
- **Unique relevant categorical samples:** **324** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `outdoor_walking` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_morning_rest`, `home_daytime_rest` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `outdoor` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### LONG_OUTDOOR_WALK — Long outdoor walk

- **Category:** `exercise`
- **Allowed `state_current` values:** `outdoor_walking`
- **Allowed `precondition` values:** `outdoor_walking`
- **Unique relevant categorical samples:** **162** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `outdoor_walking` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `outdoor_walking` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `outdoor` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### CYCLING — Cycling

- **Category:** `exercise`
- **Allowed `state_current` values:** `outdoor_cycling`, `commuting_cycle_out`, `commuting_cycle_home`
- **Allowed `precondition` values:** `commuting_cycle_out`, `commuting_cycle_home`, `outdoor_cycling`
- **Unique relevant categorical samples:** **864** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `outdoor_cycling`, `commuting_cycle_out`, `commuting_cycle_home` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_cycle_out`, `commuting_cycle_home`, `outdoor_cycling` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `morning`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `unknown` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `en_route`, `outdoor`, `unknown` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_pocket`, `in_use`, `unknown` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### LATE_NIGHT_NOISY — Late night noisy venue

- **Category:** `social`
- **Allowed `state_current` values:** `unknown_noisy`, `at_social`
- **Allowed `precondition` values:** `home_evening`, `at_social`, `at_restaurant_dinner`
- **Unique relevant categorical samples:** **4,212** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `unknown_noisy`, `at_social` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_evening`, `at_social`, `at_restaurant_dinner` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `unknown` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `social`, `unknown` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_pocket`, `in_use`, `unknown` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal` | 2 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### EDUCATION_WAITING — Waiting at tutoring

- **Category:** `education`
- **Allowed `state_current` values:** `at_education`, `at_education_class`, `at_education_break`
- **Allowed `precondition` values:** `commuting_walk_out`, `commuting_transit_out`, `at_education`
- **Unique relevant categorical samples:** **5,760** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_education`, `at_education_class`, `at_education_break` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_walk_out`, `commuting_transit_out`, `at_education` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `education` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### EDUCATION_LONG_SIT — Long sitting at tutoring

- **Category:** `education`
- **Allowed `state_current` values:** `at_education`, `at_education_class`
- **Allowed `precondition` values:** `at_education`, `at_education_class`
- **Unique relevant categorical samples:** **3,072** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_education`, `at_education_class` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `at_education`, `at_education_class` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `education` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### HOME_LUNCH — Home lunch

- **Category:** `dining`
- **Allowed `state_current` values:** `home_daytime_workday`, `home_daytime_rest`
- **Allowed `precondition` values:** `home_daytime_workday`, `home_daytime_rest`
- **Unique relevant categorical samples:** **2,880** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_daytime_workday`, `home_daytime_rest` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_daytime_workday`, `home_daytime_rest` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon`, `lunch` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### AFTER_LUNCH_WALK — After lunch walk

- **Category:** `health`
- **Allowed `state_current` values:** `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying`
- **Allowed `precondition` values:** `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying`, `home_daytime_workday`
- **Unique relevant categorical samples:** **7,680** (from `scenario_unique_samples_count.md`)
- **Substate note:** if a selected state requires light refinement, valid trigger values include `dark`.

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying`, `home_daytime_workday` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon`, `lunch` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dark`, `dim`, `normal`, `bright` | 4 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### COMMUTE_MORNING — Morning commute arrive at work

- **Category:** `commute`
- **Allowed `state_current` values:** `office_arriving`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_workday_lying`, `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out`
- **Unique relevant categorical samples:** **2,880** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_arriving` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_morning_workday`, `home_morning_workday_lying`, `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `dawn`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `transit`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `transit`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### COMMUTE_EVENING — Evening commute arrive home

- **Category:** `commute`
- **Allowed `state_current` values:** `home_evening`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy`
- **Allowed `precondition` values:** `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home`
- **Unique relevant categorical samples:** **5,888** (from `scenario_unique_samples_count.md`)
- **Substate note:** if a selected state requires light refinement, valid trigger values include `dark`.

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_evening`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `evening`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dark`, `dim`, `normal` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### HOME_TO_GYM — Arrive gym from home

- **Category:** `exercise`
- **Allowed `state_current` values:** `at_gym`, `at_gym_exercising`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_rest`, `home_daytime_workday`, `home_daytime_rest`, `home_evening`
- **Unique relevant categorical samples:** **3,240** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_gym`, `at_gym_exercising` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_morning_workday`, `home_morning_rest`, `home_daytime_workday`, `home_daytime_rest`, `home_evening` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `running`, `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `gym` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `running`, `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### DELIVERY_AT_OFFICE — Package arrived at office

- **Category:** `delivery`
- **Allowed `state_current` values:** `office_arriving`, `office_working`, `office_working_focused`, `office_working_noisy`
- **Allowed `precondition` values:** `office_working`, `office_arriving`
- **Unique relevant categorical samples:** **2,432** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_arriving`, `office_working`, `office_working_focused`, `office_working_noisy` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_working`, `office_arriving` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `forenoon`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `1` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### MEETING_UPCOMING — Meeting starting soon

- **Category:** `work`
- **Allowed `state_current` values:** `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet`
- **Unique relevant categorical samples:** **4,864** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon`, `lunch`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `cafe`, `home`, `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_down`, `face_up`, `holding_lying`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### IN_MEETING — In meeting

- **Category:** `work`
- **Allowed `state_current` values:** `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet`
- **Unique relevant categorical samples:** **2,944** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon`, `lunch`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `1` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `cafe`, `home`, `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_down`, `face_up`, `holding_lying`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### REMOTE_MEETING — Remote meeting departure

- **Category:** `commute`
- **Allowed `state_current` values:** `home_daytime_workday`, `at_cafe_quiet`
- **Allowed `precondition` values:** `home_daytime_workday`, `at_cafe_quiet`
- **Unique relevant categorical samples:** **24,576** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_daytime_workday`, `at_cafe_quiet` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_daytime_workday`, `at_cafe_quiet` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `forenoon`, `lunch`, `morning`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `stationary`, `transit`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `airport`, `cafe`, `custom`, `education`, `en_route`, `gym`, `health`, `home`, `metro`, `outdoor`, `rail_station`, `restaurant`, `shopping`, `social`, `transit`, `work` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `cafe`, `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `stationary`, `transit`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### NO_MEETINGS — No meetings today

- **Category:** `work`
- **Allowed `state_current` values:** `office_arriving`, `office_working`
- **Allowed `precondition` values:** `office_arriving`, `office_working`, `home_daytime_workday`
- **Unique relevant categorical samples:** **2,880** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_arriving`, `office_working` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_arriving`, `office_working`, `home_daytime_workday` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `forenoon`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### CAL_HAS_EVENTS — Today's schedule reminder

- **Category:** `work`
- **Allowed `state_current` values:** `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet`
- **Allowed `precondition` values:** `office_working`, `home_daytime_workday`, `at_cafe_quiet`
- **Unique relevant categorical samples:** **3,648** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_working`, `home_daytime_workday`, `at_cafe_quiet` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon`, `lunch`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `cafe`, `home`, `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_down`, `face_up`, `holding_lying`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### RETURN_OFFICE_AFTER_LUNCH — Return office after lunch

- **Category:** `work`
- **Allowed `state_current` values:** `office_working`, `office_working_focused`
- **Allowed `precondition` values:** `at_restaurant_lunch`, `at_restaurant_other`, `at_cafe`, `at_cafe_quiet`
- **Unique relevant categorical samples:** **2,304** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_working`, `office_working_focused` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `at_restaurant_lunch`, `at_restaurant_other`, `at_cafe`, `at_cafe_quiet` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### HOME_AFTER_GYM — Home after gym

- **Category:** `exercise`
- **Allowed `state_current` values:** `home_evening`, `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`
- **Allowed `precondition` values:** `at_gym`, `at_gym_exercising`
- **Unique relevant categorical samples:** **6,336** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_evening`, `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `at_gym`, `at_gym_exercising` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `lunch`, `morning`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### COMMUTE_FROM_HOME — Commuting from home

- **Category:** `commute`
- **Allowed `state_current` values:** `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_workday_lying`
- **Unique relevant categorical samples:** **180** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_morning_workday`, `home_morning_workday_lying` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `dawn`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `transit`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `normal` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `en_route` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `transit`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### OFFICE_TO_CAFE — Office to cafe

- **Category:** `dining`
- **Allowed `state_current` values:** `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `office_working_noisy`, `office_lunch_break`
- **Unique relevant categorical samples:** **9,216** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_cafe`, `at_cafe_quiet` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_working`, `office_working_focused`, `office_working_noisy`, `office_lunch_break` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `lunch`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `cafe` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal` | 2 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### LATE_NIGHT_PHONE — Late night phone too long

- **Category:** `sleep`
- **Allowed `state_current` values:** `home_sleeping_lying`
- **Allowed `precondition` values:** `home_evening`, `home_evening_lying`, `home_sleeping`
- **Unique relevant categorical samples:** **576** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_sleeping_lying` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_evening`, `home_evening_lying`, `home_sleeping` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `late_night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dark`, `dim`, `normal` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `sitting`, `sleeping` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### OFFICE_FOCUS_LONG — Long focused office session

- **Category:** `work`
- **Allowed `state_current` values:** `office_working_focused`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`
- **Unique relevant categorical samples:** **192** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_working_focused` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_working`, `office_working_focused` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### HOME_DARK_LONG — Home daytime dark too long

- **Category:** `health`
- **Allowed `state_current` values:** `home_daytime_workday_dark`, `home_daytime_rest_dark`
- **Allowed `precondition` values:** `home_daytime_workday_dark`, `home_daytime_rest_dark`, `home_evening_dark`
- **Unique relevant categorical samples:** **2,160** (from `scenario_unique_samples_count.md`)
- **Substate note:** if a selected state requires light refinement, valid trigger values include `dark`.

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_daytime_workday_dark`, `home_daytime_rest_dark` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_daytime_workday_dark`, `home_daytime_rest_dark`, `home_evening_dark` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon`, `lunch` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dark`, `dim`, `normal`, `bright` | 4 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### UNKNOWN_LONG_STAY — Long stay at unknown place

- **Category:** `other`
- **Allowed `state_current` values:** `stationary_unknown`, `unknown_noisy`, `unknown_dark`, `unknown_settled`, `unknown_lying`
- **Allowed `precondition` values:** `stationary_unknown`, `unknown_settled`
- **Unique relevant categorical samples:** **2,160** (from `scenario_unique_samples_count.md`)
- **Substate note:** if a selected state requires light refinement, valid trigger values include `dark`.

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `stationary_unknown`, `unknown_noisy`, `unknown_dark`, `unknown_settled`, `unknown_lying` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `stationary_unknown`, `unknown_settled` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `unknown` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `unknown` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `unknown` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `unknown` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dark`, `dim`, `normal`, `bright` | 4 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `sitting`, `standing`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### OFFICE_LONG_SESSION — Extra long office session

- **Category:** `work`
- **Allowed `state_current` values:** `office_working`, `office_working_focused`, `office_working_noisy`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `office_working_noisy`
- **Unique relevant categorical samples:** **2,208** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `office_working`, `office_working_focused`, `office_working_noisy` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `office_working`, `office_working_focused`, `office_working_noisy` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `forenoon` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `work` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### HOME_EVENING_DARK — Home evening dark

- **Category:** `home`
- **Allowed `state_current` values:** `home_evening_dark`
- **Allowed `precondition` values:** `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `home_evening`
- **Unique relevant categorical samples:** **2,304** (from `scenario_unique_samples_count.md`)
- **Substate note:** if a selected state requires light refinement, valid trigger values include `dark`.

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_evening_dark` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `home_evening` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `evening`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dark`, `dim`, `normal` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### HOME_EVENING_NOISY — Home evening noisy

- **Category:** `home`
- **Allowed `state_current` values:** `home_evening_noisy`
- **Allowed `precondition` values:** `home_evening`, `at_social`, `at_restaurant_dinner`
- **Unique relevant categorical samples:** **864** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_evening_noisy` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_evening`, `at_social`, `at_restaurant_dinner` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `evening`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal` | 2 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### SCHOOL_QUIET — School quiet time

- **Category:** `education`
- **Allowed `state_current` values:** `at_education_class`
- **Allowed `precondition` values:** `at_education`, `at_education_class`
- **Unique relevant categorical samples:** **768** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_education_class` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `at_education`, `at_education_class` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `education` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### CAFE_QUIET — Quiet cafe

- **Category:** `dining`
- **Allowed `state_current` values:** `at_cafe_quiet`
- **Allowed `precondition` values:** `at_cafe`
- **Unique relevant categorical samples:** **768** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `at_cafe_quiet` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `at_cafe` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `lunch`, `night` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `cafe` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal` | 2 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### TRAIN_DEPARTURE — Train departure reminder

- **Category:** `travel`
- **Allowed `state_current` values:** `commuting_transit_out`, `at_transit_hub`, `at_rail_station`
- **Allowed `precondition` values:** `commuting_transit_out`, `at_transit_hub`, `at_rail_station`
- **Unique relevant categorical samples:** **3,348** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `commuting_transit_out`, `at_transit_hub`, `at_rail_station` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_transit_out`, `at_transit_hub`, `at_rail_station` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `transit`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `1` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `en_route`, `rail_station`, `transit` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `transit`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### FLIGHT_BOARDING — Flight boarding reminder

- **Category:** `travel`
- **Allowed `state_current` values:** `commuting_drive_out`, `commuting_transit_out`, `at_airport`
- **Allowed `precondition` values:** `commuting_drive_out`, `commuting_transit_out`, `at_airport`
- **Unique relevant categorical samples:** **1,836** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `commuting_drive_out`, `commuting_transit_out`, `at_airport` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_drive_out`, `commuting_transit_out`, `at_airport` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `driving`, `stationary`, `transit`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `1` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `airport`, `en_route` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `driving`, `stationary`, `transit`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### HOTEL_CHECKIN — Hotel check-in reminder

- **Category:** `travel`
- **Allowed `state_current` values:** `commuting_drive_out`, `commuting_transit_out`, `at_airport`, `at_transit_hub`
- **Allowed `precondition` values:** `commuting_drive_out`, `commuting_transit_out`, `at_airport`, `at_transit_hub`
- **Unique relevant categorical samples:** **4,608** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `commuting_drive_out`, `commuting_transit_out`, `at_airport`, `at_transit_hub` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_drive_out`, `commuting_transit_out`, `at_airport`, `at_transit_hub` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `driving`, `stationary`, `transit`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `1` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `airport`, `en_route`, `transit` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `none` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `driving`, `stationary`, `transit`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### MOVIE_TICKET — Movie ticket reminder

- **Category:** `entertainment`
- **Allowed `state_current` values:** `home_evening`, `at_social`, `at_restaurant_other`
- **Allowed `precondition` values:** `home_evening`, `at_social`, `at_restaurant_other`
- **Unique relevant categorical samples:** **14,976** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `home_evening`, `at_social`, `at_restaurant_other` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `home_evening`, `at_social`, `at_restaurant_other` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet`, `silent` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `1` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `home`, `restaurant`, `social` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `charging`, `face_up`, `holding_lying`, `in_pocket`, `in_use`, `on_desk` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal` | 2 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0`, `1` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### HOSPITAL_APPOINTMENT — Hospital appointment reminder

- **Category:** `health`
- **Allowed `state_current` values:** `commuting_drive_out`, `commuting_transit_out`, `at_health`
- **Allowed `precondition` values:** `commuting_drive_out`, `commuting_transit_out`, `at_health`
- **Unique relevant categorical samples:** **2,052** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `commuting_drive_out`, `commuting_transit_out`, `at_health` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `commuting_drive_out`, `commuting_transit_out`, `at_health` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `0` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[0,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `1` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `en_route`, `health` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `holding_lying`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `cycling`, `driving`, `running`, `stationary`, `transit`, `unknown`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing`, `unknown` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

### RIDESHARE_PICKUP — Rideshare pickup reminder

- **Category:** `travel`
- **Allowed `state_current` values:** `outdoor_walking`, `at_social`, `at_transit_hub`
- **Allowed `precondition` values:** `outdoor_walking`, `at_social`, `at_transit_hub`
- **Unique relevant categorical samples:** **4,320** (from `scenario_unique_samples_count.md`)

| feature | dependency | possible values | alignment note |
| --- | --- | --- | --- |
| `state_current` | state_code anchor | `outdoor_walking`, `at_social`, `at_transit_hub` | Must be one of the scenario-valid StateCode values. |
| `precondition` | rule dependent | `outdoor_walking`, `at_social`, `at_transit_hub` | Chosen from the rule precondition list, or `none` if the scenario has no explicit precondition. |
| `state_duration_sec` | state_code dependent | `>=600 sec when chosen state_current is a substate; otherwise >=0 sec` | If the chosen `state_current` is a substate, the duration must support sustained refinement. |
| `ps_time` | state_code dependent | `afternoon`, `evening`, `forenoon`, `morning` | Must align with the chosen `state_current`; rule hour windows narrow this further when present. |
| `hour` | state_code / rule dependent | `state_code dependent / scenario-compatible` | Must remain consistent with `ps_time` and any explicit hour rule. |
| `cal_hasUpcoming` | rule dependent | `1` | Calendar feature driven by explicit rule logic or scenario intent. |
| `ps_dayType` | state_code dependent | `holiday`, `weekend`, `workday` | Must align with the chosen `state_current`. |
| `ps_motion` | state_code dependent | `stationary`, `walking` | Must align with the chosen `state_current`. |
| `wifiLost` | rule dependent | `0` | Explicit trigger field for departure/transition scenarios. |
| `wifiLostCategory` | rule dependent | `unknown` | Explicit trigger field when used in the rule. |
| `cal_eventCount` | rule dependent | `[1,4]` | Keep compatible with `cal_hasUpcoming`. |
| `cal_inMeeting` | rule dependent | `0` | Explicit calendar-driven signal. |
| `cal_nextLocation` | rule dependent | `unknown` | Mapped calendar next-location category. |
| `ps_sound` | state_code dependent (substate-sensitive) | `noisy`, `normal`, `quiet` | Must satisfy the chosen substate when the state is sound-refined; otherwise remain scenario-compatible. |
| `sms_delivery_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_train_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_flight_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hotel_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_movie_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_hospital_pending` | rule dependent | `0` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `sms_ride_pending` | rule dependent | `1` | Specific booking/entity feature; usually `0` unless the scenario explicitly uses it. |
| `timestep` | state_code / rule aligned | `derived from allowed hour/time window` | Fine-grained second-of-day signal compatible with the scenario time window. |
| `ps_location` | state_code dependent | `outdoor`, `social`, `transit` | Must align with the chosen `state_current`. |
| `ps_phone` | state_code dependent (substate-sensitive) | `face_up`, `in_pocket`, `in_use` | Must satisfy the chosen substate when the state is phone-refined; otherwise remain scenario-compatible. |
| `ps_light` | state_code dependent (substate-sensitive) | `dim`, `normal`, `bright` | 3 | Must satisfy the chosen substate when the state is light-refined; otherwise remain scenario-compatible. |
| `batteryLevel` | free context | `[35,95]` | Not determined by `state_current`; only needs to remain realistic. |
| `isCharging` | state_code aligned | `0` | Usually free, but must align with charging-trigger substates such as `unknown_settled`. |
| `networkType` | state_code aligned | `cellular`, `wifi` | Should remain realistic for the chosen location/context. |
| `transportMode` | state_code aligned | `stationary`, `walking` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityState` | state_code aligned | `active`, `sitting`, `standing` | Should be consistent with `ps_motion` and the chosen `state_current`. |
| `activityDuration` | state_code aligned | `state/current-context dependent >=0` | Must be realistic for the chosen activity/state combination. |
| `user_id_hash_bucket` | profile | `all 32 buckets` | Independently sampled profile feature. |
| `age_bucket` | profile | `all 8 buckets` | Independently sampled profile feature. |
| `sex` | profile | `all 4 classes` | Independently sampled profile feature. |
| `has_kids` | profile | `0`, `1` | Independently sampled profile feature. |

## Appendix A. State-code-dependent inputs

The following v0 inputs should be treated as jointly constrained during sample generation:

- `state_current`
- `state_duration_sec`
- `ps_time`
- `hour`
- `ps_dayType`
- `ps_motion`
- `ps_location`
- `ps_phone`
- `ps_sound`
- `transportMode`
- `activityState`

## Appendix B. Canonical StateCode values

`home_sleeping`, `home_morning_workday`, `home_morning_rest`, `home_daytime_workday`, `home_daytime_rest`, `home_evening`, `home_active`, `office_arriving`, `office_lunch_break`, `office_working`, `office_overtime`, `office_late_overtime`, `office_rest_day`, `commuting_walk_out`, `commuting_walk_home`, `commuting_cycle_out`, `commuting_cycle_home`, `commuting_drive_out`, `commuting_drive_home`, `commuting_transit_out`, `commuting_transit_home`, `driving`, `in_transit`, `at_metro`, `at_rail_station`, `at_airport`, `at_transit_hub`, `outdoor_walking`, `outdoor_running`, `outdoor_cycling`, `outdoor_resting`, `at_restaurant_lunch`, `at_restaurant_dinner`, `at_restaurant_other`, `at_cafe`, `at_gym_exercising`, `at_gym`, `at_shopping`, `at_health`, `at_social`, `at_education`, `at_custom`, `stationary_unknown`, `walking_unknown`, `home_sleeping_lying`, `home_morning_workday_lying`, `home_morning_rest_lying`, `home_daytime_workday_dark`, `home_daytime_workday_lying`, `home_daytime_rest_dark`, `home_daytime_rest_lying`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy`, `office_working_focused`, `office_working_noisy`, `at_cafe_quiet`, `at_education_class`, `at_education_break`, `at_health_inpatient`, `unknown_noisy`, `unknown_dark`, `unknown_settled`, `unknown_lying`

## Appendix C. LocationCategory values

`home`, `work`, `restaurant`, `cafe`, `gym`, `metro`, `rail_station`, `airport`, `transit`, `shopping`, `outdoor`, `health`, `social`, `education`, `custom`, `en_route`, `unknown`
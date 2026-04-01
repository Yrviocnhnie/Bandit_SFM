# Final Feature Space Report for the Bandit Agent

## 1. Scope and versioning

This updated report makes the **state-code dependency explicit**. Whenever labels are assigned for a sample, all `state_code dependent` features must align with the chosen `state_current` according to the StateEncoder base-state rules and substate refinement rules from the data specification.

- **v0 = initial offline training**: predict the default target recommendation from independent scenario snapshots.
- **v1 = online user-feedback learning**: adapt recommendations using short history, recent accepted actions, and feedback.

## 2. Core consistency rule for data generation

When a row is generated, the assignment order should be:

1. Choose a valid `state_current` that satisfies the scenario rule.
2. If the scenario has a `precondition`, choose a valid `precondition` from the rule list.
3. Assign all **state_code dependent** features so they are consistent with the chosen `state_current`.
4. If `state_current` is a refined substate, also satisfy the corresponding substate trigger (for example `phone=face_down`, `sound=quiet`, `sound=noisy`, `light=dark`, or `phone=charging`) and keep `state_duration_sec` compatible with the sustained-refinement rule.
5. Then assign the remaining rule-dependent and free-context features.

In other words, `state_current` is not just another categorical field. It constrains several other inputs.

## 3. State-code dependency categories used in the input space

- **state_code anchor**: the feature itself is the chosen StateCode.
- **state_code dependent**: directly determined by the base-state encoding logic.
- **state_code dependent (substate-sensitive)**: usually flexible, but forced when the chosen StateCode is a refined substate.
- **state_code aligned**: not directly encoded by StateCode, but should still be realistic given the chosen StateCode.
- **rule dependent**: driven primarily by explicit rule conditions.
- **free context**: not determined by StateCode; only needs to remain realistic.

## 4. Updated input feature table

| group | feature | v0 | v1 | feature type | feature classes | encoding | dim | assignment category | label-assignment note |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| Occurs in rules | `state_current` | Yes | Yes | categorical | 64 canonical StateCode values | one-hot | 64 | state_code anchor | Primary state-code label. All dependent features must be consistent with the chosen StateCode base rule and substate rule. |
| Occurs in rules | `precondition` | Yes | Yes | categorical | 64 StateCode values + `none` | one-hot | 65 | rule / prior-state dependent | Compact prior-state summary used when a scenario has an explicit precondition. |
| Occurs in rules | `state_duration_sec` | Yes | Yes | numeric | seconds >= 0 | 1 normalized scalar | 1 | state_code dependent | If the chosen state_current is a refined substate, this should be compatible with the >=10 minute refinement rule. |
| Occurs in rules | `ps_time` | Yes | Yes | categorical | `sleeping`, `dawn`, `morning`, `forenoon`, `lunch`, `afternoon`, `evening`, `night`, `late_night` | one-hot | 9 | state_code dependent | Must align with the base-state encoding condition of the chosen StateCode. |
| Occurs in rules | `hour` | Yes | Yes | categorical | `0..23` | one-hot | 24 | state_code / rule dependent | Must agree with ps_time and any explicit hour rules. |
| Occurs in rules | `cal_hasUpcoming` | Yes | Yes | binary | `{0,1}` | binary int | 1 | rule dependent | Driven by calendar-related scenario rules, not by StateCode. |
| Occurs in rules | `ps_dayType` | Yes | Yes | categorical | `workday`, `weekend`, `holiday` | one-hot | 3 | state_code dependent | Must align with the base-state encoding condition of the chosen StateCode. |
| Occurs in rules | `ps_motion` | Yes | Yes | categorical | `stationary`, `walking`, `running`, `cycling`, `driving`, `transit`, `unknown` | one-hot | 7 | state_code dependent | Must align with the base-state encoding condition of the chosen StateCode. |
| Occurs in rules | `wifiLost` | Yes | Yes | binary | `{0,1}` | binary int | 1 | rule dependent | Explicit trigger field used by departure/transition scenarios. |
| Occurs in rules | `wifiLostCategory` | Yes | Yes | categorical | 17 LocationCategory classes | one-hot | 17 | rule dependent | Should match explicit rule conditions when present. |
| Occurs in rules | `cal_eventCount` | Yes | Yes | numeric | count >= 0 | 1 normalized scalar | 1 | rule dependent | Calendar-context feature. |
| Occurs in rules | `cal_inMeeting` | Yes | Yes | binary | `{0,1}` | binary int | 1 | rule dependent | Calendar-context feature. |
| Occurs in rules | `cal_nextLocation` | Yes | Yes | categorical | 17 LocationCategory classes | one-hot | 17 | rule dependent | Mapped calendar next-location category. |
| Occurs in rules | `ps_sound` | Yes | Yes | categorical | `silent`, `quiet`, `normal`, `noisy`, `unknown` | one-hot | 5 | state_code dependent (substate-sensitive) | Must satisfy the chosen substate if the StateCode is sound-refined, otherwise remain scenario-compatible. |
| Occurs in rules | `sms_delivery_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | rule dependent | Specific booking/entity signal. |
| Occurs in rules | `sms_train_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | rule dependent | Specific booking/entity signal. |
| Occurs in rules | `sms_flight_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | rule dependent | Specific booking/entity signal. |
| Occurs in rules | `sms_hotel_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | rule dependent | Specific booking/entity signal. |
| Occurs in rules | `sms_movie_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | rule dependent | Specific booking/entity signal. |
| Occurs in rules | `sms_hospital_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | rule dependent | Specific booking/entity signal. |
| Occurs in rules | `sms_ride_pending` | Yes | Yes | binary | `{0,1}` | binary int | 1 | rule dependent | Specific booking/entity signal. |
| Auxiliary | `timestep` | Yes | Yes | numeric | integer in `[0,86400]` | 1 normalized scalar | 1 | state_code / rule aligned | Fine-grained time-of-day signal; should remain compatible with chosen ps_time/hour/state_current. |
| Auxiliary | `ps_location` | Yes | Yes | categorical | 17 LocationCategory classes | one-hot | 17 | state_code dependent | Must align with the base-state encoding condition of the chosen StateCode. |
| Auxiliary | `ps_phone` | Yes | Yes | categorical | `in_use`, `holding_lying`, `on_desk`, `face_up`, `in_pocket`, `face_down`, `charging`, `unknown` | one-hot | 8 | state_code dependent (substate-sensitive) | Must satisfy the chosen substate if the StateCode is phone-refined; otherwise remain scenario-compatible. |
| Auxiliary | `batteryLevel` | Yes | Yes | numeric | `0..100` | 1 normalized scalar | 1 | free context | Not determined by StateCode; should just remain realistic. |
| Auxiliary | `isCharging` | Yes | Yes | binary | `{0,1}` | binary int | 1 | state_code aligned / free context | Usually free, but should align with charging-trigger substates such as `unknown_settled`. |
| Auxiliary | `networkType` | Yes | Yes | categorical | `wifi`, `cellular`, `none` | one-hot | 3 | state_code aligned | Not directly encoded by StateCode, but should remain realistic for the chosen location/context. |
| Auxiliary | `transportMode` | Yes | Yes | categorical | `walking`, `running`, `cycling`, `driving`, `transit`, `stationary`, `unknown` | one-hot | 7 | state_code aligned | Should be consistent with the chosen StateCode and ps_motion. |
| Auxiliary | `activityState` | Yes | Yes | categorical | `sitting`, `sleeping`, `standing`, `active`, `unknown` | one-hot | 5 | state_code aligned | Should be consistent with the chosen StateCode and ps_motion. |
| Auxiliary | `activityDuration` | Yes | Yes | numeric | seconds >= 0 | 1 normalized scalar | 1 | state_code aligned | Should be realistic for the chosen activity and StateCode. |
| History (v1 only) | `hist_state_t-1` | No | Yes | categorical | 64 StateCode values + `none` | one-hot | 65 | state_code history | Recent StateCode history. |
| History (v1 only) | `hist_state_t-2` | No | Yes | categorical | 64 StateCode values + `none` | one-hot | 65 | state_code history | Recent StateCode history. |
| History (v1 only) | `hist_state_t-3` | No | Yes | categorical | 64 StateCode values + `none` | one-hot | 65 | state_code history | Recent StateCode history. |
| History (v1 only) | `prev_location` | No | Yes | categorical | 17 LocationCategory classes | one-hot | 17 | state_code aligned history | Previous location summary. |
| History (v1 only) | `prev_activityState` | No | Yes | categorical | `sitting`, `sleeping`, `standing`, `active`, `unknown` | one-hot | 5 | state_code aligned history | Previous activity summary. |
| History (v1 only) | `recent_accepted_ro_action_in_scenario` | No | Yes | categorical | 167 scenario-scoped R/O actions + `none` | one-hot | 168 | feedback dependent | Online personalization memory. |
| History (v1 only) | `recent_accepted_app_category_in_scenario` | No | Yes | categorical | 10 app categories + `none` | one-hot | 11 | feedback dependent | Online personalization memory. |
| History (v1 only) | `recent_ro_feedback_score_same_scenario` | No | Yes | numeric | `[-1,1]` | 1 scalar | 1 | feedback dependent | Online reward summary. |
| History (v1 only) | `recent_app_feedback_score_same_scenario` | No | Yes | numeric | `[-1,1]` | 1 scalar | 1 | feedback dependent | Online reward summary. |
| User profile | `user_id_hash_bucket` | Yes | Yes | categorical | 32 buckets: `b0..b31` | one-hot | 32 | profile | Stable anonymized user identity proxy. |
| User profile | `age_bucket` | Yes | Yes | categorical | 8 buckets | one-hot | 8 | profile | Broad demographic bucket. |
| User profile | `sex` | Yes | Yes | categorical | 4 classes | one-hot | 4 | profile | Optional profile attribute. |
| User profile | `has_kids` | Yes | Yes | binary | `{0,1}` | binary int | 1 | profile | Family-routine signal. |

## 5. Explicit state-code dependency notes

- `ps_location`, `ps_motion`, `ps_time`, and `ps_dayType` are **base-state dependent**.
- `ps_phone` and `ps_sound` are **substate-sensitive**. They only become fixed when the selected `state_current` is a phone/sound-refined substate.
- `transportMode` and `activityState` should remain **aligned** with `state_current` and `ps_motion`.
- `state_duration_sec` must be compatible with substate persistence whenever a refined substate is used.
- `precondition` is not derived from `state_current`; it is a summarized prior-state field chosen from the scenario rule.

## 6. Final dimensions

- **v0 input shape**: `(312,)`
- **v1 input shape**: `(710,)`

## 7. Practical recommendation for future sample generation

Any scenario-level value table or generator should treat the following features as jointly constrained:

`state_current`, `state_duration_sec`, `ps_time`, `hour`, `ps_dayType`, `ps_motion`, `ps_location`, `ps_phone`, `ps_sound`, `transportMode`, `activityState`

Those fields should not be sampled independently.

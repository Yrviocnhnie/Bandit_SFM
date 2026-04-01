# Final Feature Space Report for the Bandit Agent

## 1. Scope and versioning

This updated report incorporates one additional important input feature required by the StateEncoder substate rules: **`ps_light`**.

- **v0 = initial offline training**: predict the default target recommendation from independent scenario snapshots.
- **v1 = online user-feedback learning**: adapt recommendations using short history, recent accepted actions, and feedback.

## 2. Why `ps_light` must be added

The state encoder refines several base states into substates using **Phone / Light / Sound** triggers that must be sustained for at least 10 minutes. The current input space already included `ps_phone`, `ps_sound`, and `state_duration_sec`, but it was missing **`ps_light`**.

The missing important substate-support feature is therefore:
- **`ps_light`** with classes `dark, dim, normal, bright`

This is needed for states such as:
- `home_daytime_workday_dark`
- `home_daytime_rest_dark`
- `home_evening_dark`
- `unknown_dark`

No other new subrule feature needs to be added right now, because:
- phone-trigger substates are already covered by `ps_phone`
- sound-trigger substates are already covered by `ps_sound`
- the sustained-refinement condition is already covered by `state_duration_sec`

## 3. Updated assignment categories

- **state_code anchor**: the feature itself is the chosen StateCode.
- **state_code dependent**: directly determined by the base-state encoding logic.
- **state_code dependent (substate-sensitive)**: flexible for base states, but forced when the chosen StateCode is a refined substate.
- **state_code aligned**: not directly encoded by StateCode, but should still be realistic given the chosen StateCode.
- **rule dependent**: driven primarily by explicit scenario-rule conditions.
- **free context**: not determined by StateCode; only needs to remain realistic.

## 4. Updated input feature table

| group | feature | v0 | v1 | feature type | feature classes | encoding | dim | assignment category | label-assignment note |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| Occurs in rules | `state_current` | Yes | Yes | categorical | 64 canonical StateCode values | one-hot | 64 | state_code anchor | Primary state label. |
| Occurs in rules | `precondition` | Yes | Yes | categorical | 64 StateCode values + `none` | one-hot | 65 | rule / prior-state dependent | Compact prior-state summary. |
| Occurs in rules | `state_duration_sec` | Yes | Yes | numeric | seconds >= 0 | 1 normalized scalar | 1 | state_code dependent | Must support substate refinement when needed. |
| Occurs in rules | `ps_time` | Yes | Yes | categorical | TimeSlot (9) | one-hot | 9 | state_code dependent | Must align with base-state rule. |
| Occurs in rules | `hour` | Yes | Yes | categorical | 0..23 | one-hot | 24 | state_code / rule dependent | Must agree with ps_time and hour rules. |
| Occurs in rules | `cal_hasUpcoming` | Yes | Yes | binary | {0,1} | binary int | 1 | rule dependent | Calendar-driven scenario signal. |
| Occurs in rules | `ps_dayType` | Yes | Yes | categorical | DayType (3) | one-hot | 3 | state_code dependent | Must align with base-state rule. |
| Occurs in rules | `ps_motion` | Yes | Yes | categorical | MotionCategory (7) | one-hot | 7 | state_code dependent | Must align with base-state rule. |
| Occurs in rules | `wifiLost` | Yes | Yes | binary | {0,1} | binary int | 1 | rule dependent | Departure/transition trigger. |
| Occurs in rules | `wifiLostCategory` | Yes | Yes | categorical | LocationCategory (17) | one-hot | 17 | rule dependent | Transition trigger category. |
| Occurs in rules | `cal_eventCount` | Yes | Yes | numeric | count >= 0 | 1 normalized scalar | 1 | rule dependent | Future-event density. |
| Occurs in rules | `cal_inMeeting` | Yes | Yes | binary | {0,1} | binary int | 1 | rule dependent | Meeting flag. |
| Occurs in rules | `cal_nextLocation` | Yes | Yes | categorical | LocationCategory (17) | one-hot | 17 | rule dependent | Mapped next-event location. |
| Occurs in rules | `ps_sound` | Yes | Yes | categorical | SoundCategory (5) | one-hot | 5 | state_code dependent (substate-sensitive) | Required for sound-refined substates. |
| Auxiliary | `ps_phone` | Yes | Yes | categorical | PhoneCategory (8) | one-hot | 8 | state_code dependent (substate-sensitive) | Required for phone-refined substates. |
| Auxiliary | `ps_light` | Yes | Yes | categorical | LightCategory (4) | one-hot | 4 | state_code dependent (substate-sensitive) | Required for light-refined substates. |
| Occurs in rules | `sms_*_pending` | Yes | Yes | binary | {0,1} | binary int | 7 | rule dependent | Entity/booking flags. |
| Auxiliary | `timestep` | Yes | Yes | numeric | 0..86400 seconds | 1 normalized scalar | 1 | state_code / rule aligned | Fine-grained second-of-day signal. |
| Auxiliary | `ps_location` | Yes | Yes | categorical | LocationCategory (17) | one-hot | 17 | state_code dependent | Must align with base-state rule. |
| Auxiliary | `batteryLevel` | Yes | Yes | numeric | 0..100 | 1 normalized scalar | 1 | free context | Operational device context. |
| Auxiliary | `isCharging` | Yes | Yes | binary | {0,1} | binary int | 1 | state_code aligned | Should be realistic; may be forced by charging-related substates. |
| Auxiliary | `networkType` | Yes | Yes | categorical | wifi, cellular, none | one-hot | 3 | state_code aligned | Should fit the location/context. |
| Auxiliary | `transportMode` | Yes | Yes | categorical | walking, running, cycling, driving, transit, stationary, unknown | one-hot | 7 | state_code aligned | Should align with motion and state_current. |
| Auxiliary | `activityState` | Yes | Yes | categorical | sitting, sleeping, standing, active, unknown | one-hot | 5 | state_code aligned | Should align with motion and state_current. |
| Auxiliary | `activityDuration` | Yes | Yes | numeric | seconds >= 0 | 1 normalized scalar | 1 | state_code aligned | Should be realistic for the chosen activity. |
| History (v1 only) | `hist_state_t-1` | No | Yes | categorical | 64 StateCode + none | one-hot | 65 | history | Most recent previous state. |
| History (v1 only) | `hist_state_t-2` | No | Yes | categorical | 64 StateCode + none | one-hot | 65 | history | Second previous state. |
| History (v1 only) | `hist_state_t-3` | No | Yes | categorical | 64 StateCode + none | one-hot | 65 | history | Third previous state. |
| History (v1 only) | `prev_location` | No | Yes | categorical | LocationCategory (17) | one-hot | 17 | history | Previous location summary. |
| History (v1 only) | `prev_activityState` | No | Yes | categorical | sitting, sleeping, standing, active, unknown | one-hot | 5 | history | Previous activity summary. |
| History (v1 only) | `recent_accepted_ro_action_in_scenario` | No | Yes | categorical | 167 scenario-scoped R/O actions + none | one-hot | 168 | history / personalization | Recent accepted RO action. |
| History (v1 only) | `recent_accepted_app_category_in_scenario` | No | Yes | categorical | 10 app categories + none | one-hot | 11 | history / personalization | Recent accepted app category. |
| History (v1 only) | `recent_ro_feedback_score_same_scenario` | No | Yes | numeric | [-1,1] | 1 scalar | 1 | history / personalization | Recent RO reward. |
| History (v1 only) | `recent_app_feedback_score_same_scenario` | No | Yes | numeric | [-1,1] | 1 scalar | 1 | history / personalization | Recent App reward. |
| User profile | `user_id_hash_bucket` | Yes | Yes | categorical | 32 buckets | one-hot | 32 | profile | Anonymized identity bucket. |
| User profile | `age_bucket` | Yes | Yes | categorical | 8 buckets | one-hot | 8 | profile | Age bucket. |
| User profile | `sex` | Yes | Yes | categorical | 4 buckets | one-hot | 4 | profile | Demographic feature. |
| User profile | `has_kids` | Yes | Yes | binary | {0,1} | binary int | 1 | profile | Family-routine signal. |

## 5. Updated dimension summary

| group | v0 dim | v1 dim |
| --- | ---: | ---: |
| Occurs in rules | 223 | 223 |
| Auxiliary | 48 | 48 |
| History (v1 only) | 0 | 398 |
| User profile | 45 | 45 |
| **Total** | **316** | **714** |

- **v0 input shape:** `(316,)`
- **v1 input shape:** `(714,)`

## 6. Final note on state-code alignment

When assigning labels or generating samples, `ps_location`, `ps_motion`, `ps_time`, `ps_dayType`, `ps_phone`, `ps_light`, `ps_sound`, `transportMode`, and `activityState` should not be chosen independently of `state_current`. They must be made consistent with the chosen base state and, when applicable, with the substate trigger rule.

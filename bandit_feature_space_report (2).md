# Final Feature Space Report for the Bandit Agent

## 1. Scope and naming

This report defines the **final shared feature space** for the bandit agent after the latest updates.

### Training phases

- **v0 = initial offline training**
  - Goal: predict the current **default target recommendation** from the labeled scenario files.
  - Assumption: training examples can be treated as mostly **independent scenario snapshots**.
  - Consequence: **history features are not required in v0**.

- **v1 = online user-feedback learning**
  - Goal: personalize the recommendation policy using likes, dislikes, accepts, dismissals, and recent behavior.
  - Consequence: **history and feedback features are included in v1**.

The same overall feature taxonomy is used for both the **RO model** and the **App model**, but the active subset differs by phase.

---

## 2. Verified files and consistency

- `default_rules_1_english.json` contains **65 scenarios / rules**.
- `scenario_recommendation_actions_v4_english.md` contains **65 scenarios** with recommendation lists.
- The rules file and the v4 recommendation file now align at **65 / 65 scenarios**.
- The output space for the current report is therefore based on the **full 65-scenario recommendation file**.

### Important schema note

The rules use the following SMS-derived fields:

- `sms_delivery_pending`
- `sms_train_pending`
- `sms_flight_pending`
- `sms_hotel_pending`
- `sms_movie_pending`
- `sms_hospital_pending`
- `sms_ride_pending`

If any of these are missing from the current runtime schema / DataTray pipeline, they should be added before full end-to-end training and serving.

---

## 3. Design decision for `scenarioId`

### Final decision

Yes — **`scenarioId` should be part of the feature-space input**.

This is reasonable because your current objective is:
- first, detect / match the scenario,
- then predict the correct default recommendation for that scenario,
- and later personalize inside that scenario using user feedback.

So in the final design:

- `scenarioId` is included as an **explicit input feature**.
- `scenarioId` is also still useful as a **candidate-mask / serving-control variable**, because each scenario has its own valid action list.

### Why this makes sense

Including `scenarioId` helps the model learn the baseline mapping from scenario to the default target recommendation. That is especially appropriate in **v0**, where the goal is to reproduce the existing default policy.

At the same time, the agent should **not rely only on `scenarioId`**. The rest of the context features remain important because:
- they support robustness when scenarios are noisy or imperfectly matched,
- they support future online learning,
- they allow the model to adapt within the same scenario using user profile and feedback.

---

## 4. Classes / taxonomies of the main features

This section lists the main feature classes and their intended role.

### 4.1 Scenario control

| feature | classes / range | rationale |
| --- | --- | --- |
| `scenarioId` | 65 scenario classes | Directly identifies which scenario the current sample belongs to. Useful for baseline default-target prediction and candidate masking. |

### 4.2 Core rule-derived context

| feature | classes / range | rationale |
| --- | --- | --- |
| `state_current` | 64 canonical StateCode classes | Main semantic summary of context; strongest feature in the rules. |
| `state_duration_sec` | scalar duration | Disambiguates short vs long dwell / long session / overtime / long transit cases. |
| `hour` | 24 classes | Useful exact hour-of-day signal used by some rules. |
| `ps_time` | 9 classes | Higher-level time-of-day bucket; better generalization than exact hour. |
| `cal_hasUpcoming` | 3 classes (`true/false/unknown`) | Needed for calendar-driven scenarios. |
| `cal_nextMinutes` | scalar | Time to next event. Important for meeting / departure timing. |
| `ps_dayType` | 3 classes | Workday / weekend / holiday split. |
| `ps_motion` | 7 classes | Physical motion pattern. |
| `wifiLost` | 3 classes (`true/false/unknown`) | Strong transition signal for leaving a place. |
| `wifiLostCategory` | 17 classes | Which type of place Wi-Fi was lost from. |
| `cal_eventCount` | scalar | Day density / number of events. |
| `cal_inMeeting` | 3 classes (`true/false/unknown`) | In-meeting state. |
| `cal_nextLocation` | 17 classes | Next calendar location using the same location taxonomy; default includes `unknown`. |
| `ps_sound` | 5 classes | Quiet / noisy context. |
| SMS pending fields | 3 classes each (`true/false/unknown`) | Needed for delivery / travel / booking / ride related scenarios. |

### 4.3 Auxiliary context

| feature | classes / range | rationale |
| --- | --- | --- |
| `timestep` | integer in `[0, 86400]` | Fine-grained second-of-day signal. Explicitly requested. |
| `ps_location` | 17 classes | Raw location category. Helps beyond compressed state codes. |
| `ps_phone` | 8 classes | Phone posture / interaction context. |
| `batteryLevel` | scalar in `[0,100]` | Useful for operations and practical device-aware recommendations. |
| `isCharging` | 3 classes (`true/false/unknown`) | Helps with home / desk / sleep / stationary contexts. |
| `networkType` | 4 classes | Useful for connectivity-sensitive operations. |
| `transportMode` | 7 classes | Stable mobility abstraction. |
| `activityState` | 5 classes | High-level activity summary. |
| `activityDuration` | scalar | Helps distinguish short vs sustained activity. |

### 4.4 History features (v1 only)

| feature | classes / range | rationale |
| --- | --- | --- |
| `hist_state_1` | 64 classes | Most recent previous state. |
| `hist_state_2` | 64 classes | Second previous state. |
| `hist_state_3` | 64 classes | Third previous state. |
| `prev_location` | 17 classes | Previous location category. |
| `prev_activityState` | 5 classes | Previous activity summary. |
| `recent_accepted_ro_action_same_scenario` | 168 classes (`167 actions + none`) | Captures recent preference for RO actions in that scenario. |
| `recent_accepted_app_category_same_scenario` | 11 classes (`10 categories + none`) | Captures recent preference for app categories in that scenario. |
| `recent_ro_feedback_score_same_scenario` | scalar in `[-1,1]` | Aggregated recent reward signal. |
| `recent_app_feedback_score_same_scenario` | scalar in `[-1,1]` | Aggregated recent reward signal. |

### 4.5 User profile

| feature | classes / range | rationale |
| --- | --- | --- |
| `user_id_hash_bucket` | 32 buckets | Stable anonymized per-user identity signal. Raw name should not be used directly. |
| `age` | scalar | Helps model broad preference differences. |
| `sex` | 4 classes | Optional demographic feature if available and permitted. |
| `kids` | 3 classes | Useful for family-related routine variation. |

---

## 5. Feature-frequency statistics from the scenario rules

All features that appear in the rules are included in the final feature space.

| feature | scenarios using it | coverage | total uses | in conditions | in preconditions |
| --- | ---: | ---: | ---: | ---: | ---: |
| state_current | 54 | 83.1% | 62 | 52 | 10 |
| state_duration_sec | 11 | 16.9% | 11 | 11 | 0 |
| ps_time | 5 | 7.7% | 5 | 5 | 0 |
| hour | 4 | 6.2% | 6 | 6 | 0 |
| cal_hasUpcoming | 3 | 4.6% | 3 | 3 | 0 |
| cal_nextMinutes | 2 | 3.1% | 2 | 2 | 0 |
| ps_dayType | 2 | 3.1% | 2 | 2 | 0 |
| ps_motion | 2 | 3.1% | 2 | 2 | 0 |
| wifiLost | 2 | 3.1% | 2 | 2 | 0 |
| wifiLostCategory | 2 | 3.1% | 2 | 2 | 0 |
| cal_eventCount | 1 | 1.5% | 1 | 1 | 0 |
| cal_inMeeting | 1 | 1.5% | 1 | 1 | 0 |
| cal_nextLocation | 1 | 1.5% | 1 | 1 | 0 |
| ps_sound | 1 | 1.5% | 1 | 1 | 0 |
| sms_delivery_pending | 1 | 1.5% | 1 | 1 | 0 |
| sms_flight_pending | 1 | 1.5% | 1 | 1 | 0 |
| sms_hospital_pending | 1 | 1.5% | 1 | 1 | 0 |
| sms_hotel_pending | 1 | 1.5% | 1 | 1 | 0 |
| sms_movie_pending | 1 | 1.5% | 1 | 1 | 0 |
| sms_ride_pending | 1 | 1.5% | 1 | 1 | 0 |
| sms_train_pending | 1 | 1.5% | 1 | 1 | 0 |

### Interpretation

- `state_current` is the dominant rule feature and should remain the backbone of the model.
- `state_duration_sec`, `ps_time`, and `hour` are the most important secondary disambiguators.
- Calendar, Wi-Fi transition, sound, and SMS fields are narrow but necessary for a small number of highly specific scenarios.

---

## 6. `state_current` usage statistics

### Coverage of the StateCode space

- Total canonical `StateCode` values in the data spec: **64**
- Canonical `StateCode` values used in the rules: **57 / 64 = 89.1%**
- Base states used: **38 / 44 = 86.4%**
- Substates used: **19 / 20 = 95.0%**

### Canonicalization note

The rules contain legacy labels:
- `at_education_quiet`
- `at_education_noisy`

These should be normalized to the current spec names:
- `at_education_class`
- `at_education_break`

### Canonical states not used by current rules

- `at_custom`
- `at_health`
- `at_health_inpatient`
- `home_active`
- `in_transit`
- `outdoor_resting`
- `walking_unknown`

### Most reused canonical `state_current` values

| state_current | scenarios using it | total mentions |
| --- | ---: | ---: |
| office_working | 9 | 9 |
| office_working_focused | 7 | 7 |
| at_cafe_quiet | 6 | 6 |
| home_daytime_rest | 6 | 6 |
| home_morning_workday | 6 | 6 |
| office_overtime | 6 | 6 |
| at_cafe | 5 | 5 |
| at_gym | 5 | 5 |
| at_gym_exercising | 5 | 5 |
| home_daytime_workday | 5 | 5 |
| home_evening | 5 | 5 |
| home_morning_rest | 5 | 5 |
| office_working_noisy | 5 | 5 |
| office_arriving | 4 | 4 |
| outdoor_walking | 4 | 4 |
| at_education_class | 3 | 3 |
| commuting_cycle_out | 3 | 3 |
| commuting_drive_home | 3 | 3 |
| commuting_drive_out | 3 | 3 |
| home_daytime_rest_dark | 3 | 3 |

### Interpretation

The rules use most of the available state space, which is good news for the bandit design: `state_current` is broad enough to carry a large amount of semantic context, not just a narrow subset.

---

## 7. Final feature space for the bandit agent

This is the final recommended shared feature space.

### Column meanings

- **group** = category of feature
- **encoding** = suggested representation
- **dim** = input dimensionality contributed by that feature
- **v0** = include in initial offline training
- **v1** = include in online user-feedback training

| feature | group | encoding | dim | v0 | v1 | rationale |
| --- | --- | --- | ---: | --- | --- | --- |
| scenarioId | scenario control | one-hot | 65 | Yes | Yes | Explicit scenario identity. Good for baseline target prediction and scenario-conditioned recommendation ranking. |
| state_current | occurs in rules | one-hot | 64 | Yes | Yes | Strongest context feature. |
| state_duration_sec | occurs in rules | 1 scalar (log1p normalized) | 1 | Yes | Yes | Long-stay / long-session disambiguation. |
| hour | occurs in rules | one-hot | 24 | Yes | Yes | Exact hour signal. |
| ps_time | occurs in rules | one-hot | 9 | Yes | Yes | High-level time-of-day slot. |
| cal_hasUpcoming | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Calendar-driven behavior. |
| cal_nextMinutes | occurs in rules | 1 scalar | 1 | Yes | Yes | Time-to-event signal. |
| ps_dayType | occurs in rules | one-hot | 3 | Yes | Yes | Workday vs weekend behavior. |
| ps_motion | occurs in rules | one-hot | 7 | Yes | Yes | Motion abstraction. |
| wifiLost | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Transition / leaving-place signal. |
| wifiLostCategory | occurs in rules | one-hot | 17 | Yes | Yes | Which place category was left. |
| cal_eventCount | occurs in rules | 1 scalar | 1 | Yes | Yes | Number of events. |
| cal_inMeeting | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | In-meeting context. |
| cal_nextLocation | occurs in rules | one-hot | 17 | Yes | Yes | Next location category, with default `unknown`. |
| ps_sound | occurs in rules | one-hot | 5 | Yes | Yes | Quiet / noisy context. |
| sms_delivery_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Delivery-related scenario. |
| sms_train_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Train-related scenario. |
| sms_flight_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Flight-related scenario. |
| sms_hotel_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Hotel-related scenario. |
| sms_movie_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Movie-related scenario. |
| sms_hospital_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Hospital-related scenario. |
| sms_ride_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Ride-related scenario. |
| timestep | auxiliary | 1 scalar in `[0, 86400]` | 1 | Yes | Yes | Fine-grained second-of-day signal. |
| ps_location | auxiliary | one-hot | 17 | Yes | Yes | Raw location category for generalization. |
| ps_phone | auxiliary | one-hot | 8 | Yes | Yes | Phone posture / use context. |
| batteryLevel | auxiliary | 1 scalar in `[0,100]` | 1 | Yes | Yes | Device readiness / practicality signal. |
| isCharging | auxiliary | one-hot true/false/unknown | 3 | Yes | Yes | Device charging context. |
| networkType | auxiliary | one-hot | 4 | Yes | Yes | Connectivity-sensitive operations. |
| transportMode | auxiliary | one-hot | 7 | Yes | Yes | Stable movement mode abstraction. |
| activityState | auxiliary | one-hot | 5 | Yes | Yes | High-level activity abstraction. |
| activityDuration | auxiliary | 1 scalar (log1p normalized) | 1 | Yes | Yes | Short vs sustained activity. |
| hist_state_1 | history | one-hot | 64 | No | Yes | Most recent previous state. Excluded from v0 because initial offline training can use independent scenario snapshots. |
| hist_state_2 | history | one-hot | 64 | No | Yes | Second most recent previous state. |
| hist_state_3 | history | one-hot | 64 | No | Yes | Third most recent previous state. |
| prev_location | history | one-hot | 17 | No | Yes | Previous location category. |
| prev_activityState | history | one-hot | 5 | No | Yes | Previous activity summary. |
| recent_accepted_ro_action_same_scenario | history / online | one-hot over `167 actions + none` | 168 | No | Yes | Recent accepted RO action in the same scenario. |
| recent_accepted_app_category_same_scenario | history / online | one-hot over `10 categories + none` | 11 | No | Yes | Recent accepted app category in the same scenario. |
| recent_ro_feedback_score_same_scenario | history / online | 1 scalar in `[-1,1]` | 1 | No | Yes | Aggregated recent RO reward signal. |
| recent_app_feedback_score_same_scenario | history / online | 1 scalar in `[-1,1]` | 1 | No | Yes | Aggregated recent App reward signal. |
| user_id_hash_bucket | user profile | one-hot hashed bucket | 32 | Yes | Yes | Stable anonymized user identity. Use hash bucket, not raw name. |
| age | user profile | 1 scalar | 1 | Yes | Yes | Broad user preference differences. |
| sex | user profile | one-hot | 4 | Yes | Yes | Optional demographic feature if permitted. |
| kids | user profile | one-hot | 3 | Yes | Yes | Family-structure signal. |

---

## 8. Features intentionally excluded or deferred

| feature | decision | reason |
| --- | --- | --- |
| `ps_light` | Excluded | You asked to remove features that are probably not useful. Current light-sensitive cases are mostly already encoded inside `state_current` dark substates. |
| `speed` | Excluded | Likely noisy and largely redundant with `ps_motion` and `transportMode`. |
| raw user name text | Excluded | Do not one-hot or embed raw names directly. Use hashed user identity instead. |
| raw latitude / longitude | Deferred | Too sparse and privacy-sensitive for the current shared bandit. |
| `wifiSsid` / geofence IDs / cell IDs | Deferred | Too device-specific and sparse for the current design. |
| extra scenario-matcher internals beyond `scenarioId` | Excluded | `scenarioId` is enough; adding more matcher internals would be redundant. |

---

## 9. Dimension summary and input shapes

### 9.1 Group dimensions

| group | dim |
| --- | ---: |
| scenario control | 65 |
| occurs in rules | 179 |
| auxiliary | 47 |
| history (state / transition) | 214 |
| history (online feedback extras) | 181 |
| user profile | 40 |

### 9.2 Shared input shapes

#### v0: initial offline training

Included groups:
- scenario control
- occurs in rules
- auxiliary
- user profile

Total dimension:

**65 + 179 + 47 + 40 = 331**

So the recommended **v0 shared input shape** is:

- single example: **`(331,)`**
- batch: **`(batch_size, 331)`**

#### v1: online user-feedback learning

Included groups:
- scenario control
- occurs in rules
- auxiliary
- history (state / transition)
- history (online feedback extras)
- user profile

Total dimension:

**65 + 179 + 47 + 214 + 181 + 40 = 726**

So the recommended **v1 shared input shape** is:

- single example: **`(726,)`**
- batch: **`(batch_size, 726)`**

### 9.3 Practical interpretation

- **v0** is intentionally simpler because it does not require constructing full realistic day trajectories.
- **v1** adds the sequential and feedback-dependent information needed for true personalization.

---

## 10. Output space and label space

### 10.1 RO model

#### v0 training target

- Use the **default RO target** defined for each of the **65 scenarios**.
- That means the current v0 default-target label space is:

**`(65,)`**

This is the simplest and most faithful offline target for reproducing the current default policy.

#### v1 online candidate space

- The full current recommendation file contains **167 R&O actions**.
- In online serving, the agent should only rank the candidate actions belonging to the matched scenario.

So for v1:
- global arm inventory: **167 R&O actions**
- actual serving candidate set: **scenario-specific masked subset**

### 10.2 App model

The App model should predict **app category**, not a concrete app.

Recommended output head:
- **10 app categories**
  - `social`
  - `productivity`
  - `entertainment`
  - `navigation`
  - `shopping`
  - `news`
  - `health`
  - `music`
  - `reading`
  - `game`

So the App model output shape is:

**`(10,)`**

### 10.3 Default app-category distribution in v4

| app category | default count |
| --- | ---: |
| navigation | 16 |
| music | 12 |
| productivity | 11 |
| social | 9 |
| shopping | 6 |
| entertainment | 3 |
| reading | 3 |
| health | 3 |
| news | 2 |

`game` is currently unused as a default label but should still remain in the category taxonomy.

---

## 11. Final verification: does this feature space make sense?

### Yes — for v0

This v0 feature space makes sense because:

1. it includes **all features that occur in the rules**,
2. it includes `scenarioId`, which is appropriate for learning the current default target recommendation,
3. it avoids requiring hard-to-build full-day history trajectories during initial offline training,
4. it still includes auxiliary and user-profile features that will remain useful later.

### Yes — for v1

This v1 feature space makes sense because:

1. it adds short-term behavioral history,
2. it adds recent feedback and accepted-action signals,
3. it supports personalization within the same scenario,
4. it uses the same base context structure as v0, so the transition from offline to online learning is clean.

### Overall judgment

This is a realistic and logically consistent feature space for your bandit pipeline:
- **v0** learns the default recommendation policy,
- **v1** personalizes that policy with user feedback.

---

## 12. Compact final spec

### Shared input vector sizes

- **v0 input:** `(331,)`
- **v1 input:** `(726,)`

### RO model

- **v0 output:** `(65,)` default RO targets
- **v1 serving arm inventory:** `167` R&O actions with per-scenario candidate masking

### App model

- **v0 / v1 output:** `(10,)` app categories

### Key implementation notes

- include `scenarioId` directly in the input vector
- keep all rule-used features in both v0 and v1
- do **not** include history features in v0
- add history + feedback features in v1
- keep `state_current` as the main semantic backbone
- use app category, not concrete app, as the App-model output
- use hashed user identity instead of raw name text
- add any missing SMS pending fields to the runtime schema before full deployment


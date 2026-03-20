# Updated Feature Space Report for the Bandit Agent (v4)

## 1. Scope and what changed

- Added `timestep` as a feature: integer seconds from midnight in the range `[0, 86400]`.

- Renamed `cal_nextLocation_type` to `cal_nextLocation` and encoded it using the same 17-class location taxonomy (default = `unknown`).

- Reorganized the final feature space into: **scenario control**, **occurs in rules**, **auxiliary**, **history**, and **user profile**.

- Recomputed the output space using **all 65 scenarios** from `scenario_recommendation_actions_v4_english.md`.

- Re-verified which rule features are actually used and how many canonical `state_current` values are covered.


## 2. Verified file consistency

- `default_rules_1_english.json` contains **65 rules / scenarios**.

- `scenario_recommendation_actions_v4_english.md` contains **65 scenarios**, **167 R&O actions**, and **1 default app recommendation per scenario**.

- The scenario ID sets in the rules file and the v4 recommendation file match exactly: **65/65**.

- The recommendation file now fully covers the rules file, so the output space can be defined over all 65 scenarios.


## 3. Important design decision: where `scenarioId` should live

The scenario should be treated as a **serving control / candidate-mask variable**, not as the main learned feature. In deployment, the rules engine or scenario matcher first identifies the scenario, and the bandit then ranks only the actions that belong to that scenario.


That means:

- `scenarioId` is still important and should be available at serving time.

- But the dense learned context vector should be built from physical state, digital context, history, and user profile.

- This avoids turning the model into a pure table lookup from scenario to default action.


## 4. Feature-frequency statistics from the scenario rules

All features that appear in the rules are included in the final design.


| feature | scenarios using it | coverage | total uses | in conditions | in preconditions |
| --- | --- | --- | --- | --- | --- |
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

- `state_current` is the dominant feature in the rules and is the backbone of the bandit context.

- The next-most-used features are `state_duration_sec`, `ps_time`, and `hour`, which act as temporal / duration disambiguators.

- Calendar, Wi-Fi-exit, sound, and SMS reminder fields are narrower but still important because they define a few highly specific scenarios.


## 5. `state_current` usage statistics

- Total canonical `StateCode` values in the data spec: **64**.

- Canonical `StateCode` values used somewhere in the rules: **57 / 64 = 89.1%**.

- Base states used: **38 / 44 = 86.4%**.

- Substates used: **19 / 20 = 95.0%**.

- Canonicalization needed: the rules use legacy labels `at_education_quiet` and `at_education_noisy`; these should be normalized to `at_education_class` and `at_education_break` to match the current data spec.

- Canonical states not used by any current rule: `at_custom, at_health, at_health_inpatient, home_active, in_transit, outdoor_resting, walking_unknown`.


### Most reused canonical `state_current` values in the rules


| state_current | scenarios using it | total mentions |
| --- | --- | --- |
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

## 6. Final feature space for the bandit agent

This section gives the **final recommended input feature space**. The same shared input vector can be used by both models:

- **Model 1:** predicts the default R/O recommendation.

- **Model 2:** predicts the default app recommendation category.


A feature is marked as **v1 = Yes** if it should be present from the first release, and **online = Yes** if it should remain or be added for later online personalization.


| feature | group | encoding | dim | v1 | online | rationale |
| --- | --- | --- | --- | --- | --- | --- |
| scenarioId | scenario control | one-hot (serving gate) | 65 | No (kept outside dense vector) | No (kept outside dense vector) | Matched scenario; used to mask candidate actions to the scenario action list. Do not rely on it as the only feature in the learned scorer. |
| state_current | occurs in rules | one-hot | 64 | Yes | Yes | Strongest feature. Main semantic summary used by the rules. |
| state_duration_sec | occurs in rules | 1 scalar (log1p normalized) | 1 | Yes | Yes | Needed for long-stay, long-drive, focus-session, and overtime cases. |
| hour | occurs in rules | one-hot | 24 | Yes | Yes | Used directly in rules; complements timestep and ps_time. |
| ps_time | occurs in rules | one-hot | 9 | Yes | Yes | High-level time slot that is easier to generalize than raw hour. |
| cal_hasUpcoming | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Meeting/schedule driven scenarios. |
| cal_nextMinutes | occurs in rules | 1 scalar | 1 | Yes | Yes | Distance to next event. |
| ps_dayType | occurs in rules | one-hot | 3 | Yes | Yes | Workday/weekend/holiday split used in rules and routines. |
| ps_motion | occurs in rules | one-hot | 7 | Yes | Yes | Captures physical movement mode. |
| wifiLost | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Important transition signal for leave-office / lunch-out style cases. |
| wifiLostCategory | occurs in rules | one-hot | 17 | Yes | Yes | Category of known place whose Wi-Fi was just lost. |
| cal_eventCount | occurs in rules | 1 scalar | 1 | Yes | Yes | Useful for day density / no-meetings style contexts. |
| cal_inMeeting | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Directly needed for in-meeting behavior. |
| cal_nextLocation | occurs in rules | one-hot | 17 | Yes | Yes | Mapped to the same 17 location classes; default unknown. |
| ps_sound | occurs in rules | one-hot | 5 | Yes | Yes | Needed for noisy/quiet contexts. |
| sms_delivery_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Package pickup scenario. |
| sms_train_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Travel reminder scenario; must be added to DataTray. |
| sms_flight_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Travel reminder scenario; must be added to DataTray. |
| sms_hotel_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Travel reminder scenario; must be added to DataTray. |
| sms_movie_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Movie-ticket scenario; must be added to DataTray. |
| sms_hospital_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Hospital appointment scenario; must be added to DataTray. |
| sms_ride_pending | occurs in rules | one-hot true/false/unknown | 3 | Yes | Yes | Rideshare pickup scenario; must be added to DataTray. |
| timestep | auxiliary | 1 scalar in [0, 86400] | 1 | Yes | Yes | Fine-grained time-of-day signal. Added per user feedback. |
| ps_location | auxiliary | one-hot | 17 | Yes | Yes | Raw location category helps generalize beyond the current rule list. |
| ps_phone | auxiliary | one-hot | 8 | Yes | Yes | Helpful for lying-down / phone posture / focus contexts. |
| batteryLevel | auxiliary | 1 scalar in [0, 100] | 1 | Yes | Yes | Useful operational context for reminders and system actions. |
| isCharging | auxiliary | one-hot true/false/unknown | 3 | Yes | Yes | Useful for late-night / desk / home contexts. |
| networkType | auxiliary | one-hot | 4 | Yes | Yes | Affects whether network-dependent operations make sense. |
| transportMode | auxiliary | one-hot | 7 | Yes | Yes | More stable mobility signal than raw speed. |
| activityState | auxiliary | one-hot | 5 | Yes | Yes | High-level activity summary that complements motion. |
| activityDuration | auxiliary | 1 scalar (log1p normalized) | 1 | Yes | Yes | Useful for sustained focus / sedentary / waiting contexts. |
| hist_state_1 | history | one-hot | 64 | Yes | Yes | Most recent previous state_current from the state chain. |
| hist_state_2 | history | one-hot | 64 | Yes | Yes | Second most recent previous state. |
| hist_state_3 | history | one-hot | 64 | Yes | Yes | Third most recent previous state. |
| prev_location | history | one-hot | 17 | Yes | Yes | Previous location category. Useful for transitions such as home→gym or office→cafe. |
| prev_activityState | history | one-hot | 5 | Yes | Yes | Previous activity summary. User explicitly requested prev_activity. |
| recent_accepted_ro_action_same_scenario | history / online | one-hot over 167 actions + none | 168 | No | Yes | Personalization signal once feedback exists. |
| recent_accepted_app_category_same_scenario | history / online | one-hot over 10 categories + none | 11 | No | Yes | Personalization signal for app-category preference. |
| recent_ro_feedback_score_same_scenario | history / online | 1 scalar in [-1,1] | 1 | No | Yes | Aggregated recent like/dislike signal for RO actions in the same scenario. |
| recent_app_feedback_score_same_scenario | history / online | 1 scalar in [-1,1] | 1 | No | Yes | Aggregated recent like/dislike signal for app recommendation in the same scenario. |
| user_id_hash_bucket | user profile | one-hot hashed bucket | 32 | Yes | Yes | Use user id / name only as a stable hashed personalization bucket, not as raw text. |
| age | user profile | 1 scalar | 1 | Yes | Yes | Supports different routine and reminder preferences by age. |
| sex | user profile | one-hot | 4 | Yes | Yes | Optional profile attribute if available and permitted. |
| kids | user profile | one-hot | 3 | Yes | Yes | Useful for schedule / travel / family-context preferences. |

## 7. Features intentionally excluded or deferred


| feature | decision | reason |
| --- | --- | --- |
| ps_light | Excluded from final vector | User asked to remove it if not useful. Current light-sensitive scenarios are already largely captured by `state_current` substates such as `home_evening_dark` and `home_daytime_*_dark`. |
| speed | Excluded from final vector | Too noisy and redundant with `ps_motion` + `transportMode` for this bandit setup. |
| raw user name text | Excluded from dense vector | Do not one-hot raw names. Use a hashed `user_id_hash_bucket` instead. |
| ps_scenario / scenario matcher outputs | Excluded from dense vector | These are downstream matcher outputs and can leak the rule engine decision. Use `scenarioId` only as a serving gate / candidate mask. |
| latitude / longitude / wifiSsid / geofence ids | Deferred | Too sparse and environment-specific for the current cold-start bandit. |

## 8. Dimension summary and input shape

| group | dim |
| --- | --- |
| occurs in rules | 179 |
| auxiliary | 47 |
| history (basic, available at launch) | 214 |
| user profile | 40 |
| v1 shared dense input total | 480 |
| history (online personalization extras) | 181 |
| v2 shared dense input total | 661 |

### Recommended shared input shapes

- **Cold-start / default model input shape (v1):** `(480,)`

- **Online-personalized model input shape (v2):** `(661,)`

- **Optional scenario gate / candidate mask:** `scenarioId` one-hot of shape `(65,)`, kept outside the dense context vector.


## 9. Output space and label space

### 9.1 R/O model

- Number of scenarios with a default R/O label: **65**.

- Default-label distribution: **50 operations** and **15 reminders**.

- **Training output shape for the current default predictor:** `(65,)` using the **65 default R/O action IDs** (one default per scenario).

- **Future online serving candidate space:** `167` R&O action IDs, with a **per-scenario action mask** because each scenario only exposes its own 2–3 candidate R/O actions.


### 9.2 App model

- The app model should predict **app category**, not a concrete app.

- The concrete app should be chosen by a downstream mapper from `app_category -> user's actual app`.

- Recommended serving / model head size: **10 app categories** (`social`, `productivity`, `entertainment`, `navigation`, `shopping`, `news`, `health`, `music`, `reading`, `game`).

- **Current observed default app categories in v4:** 9 categories (the `game` category is currently unused as a default label).

- **Training output shape for the app model:** `(10,)`.

- Note: because each current scenario only has **one default A** in the file, true within-scenario app-action ranking is still limited. Personalization on the app side will mostly happen in the downstream app-category-to-concrete-app layer until more alternative A candidates are added.


### Default app-category distribution in v4


| app category | default count |
| --- | --- |
| navigation | 16 |
| music | 12 |
| productivity | 11 |
| social | 9 |
| shopping | 6 |
| entertainment | 3 |
| reading | 3 |
| health | 3 |
| news | 2 |

## 10. Final verification: does this feature space make sense?

### Yes — for three reasons

1. **It fully respects the rules.** Every feature that appears in the rules is included.

2. **It generalizes beyond the rules.** Auxiliary fields such as `timestep`, `ps_location`, `ps_phone`, `transportMode`, and `activityState` help the bandit learn patterns the handwritten rules do not explicitly encode.

3. **It is ready for online adaptation.** The history and profile features let the model personalize after users start accepting or rejecting actions.


### Final realism checks

- `ps_light` was intentionally dropped because light-driven behavior is already captured well through the state encoder and the relevant dark-environment substates.

- `speed` was intentionally dropped because it is noisy and overlaps with `ps_motion` and `transportMode`.

- Raw names should not be fed directly into a UCB feature vector; use a stable hashed user bucket instead.

- Six SMS fields used by the rules are still missing from the current data spec and must be added to the DataTray / snapshot pipeline before the full 65-scenario setup can be trained or served consistently.


## 11. Recommended implementation notes

- Normalize all legacy state labels in the rules before training.

- Use the same shared dense context vector for both heads.

- Keep `scenarioId` as a serving mask / gating variable.

- Train the RO model on the current default R/O label first, then switch to contextual bandit feedback updates once accept / reject data is available.

- Train the app model on app category now, and keep the concrete-app choice in a downstream mapping layer.

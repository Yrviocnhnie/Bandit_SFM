# Scenario Unique Samples Count Report (Strict State/Rule Validated)

## 1. Scope

This report updates the per-scenario unique-sample counts using the **strict state/rule validated enumerator** from the latest generator. These counts supersede the earlier approximate counts wherever the previous counting policy overestimated support.

The strict counts are based on generated rows that simultaneously satisfy:
- the chosen `state_current`
- StateEncoder base-state constraints
- substate trigger constraints such as `ps_phone`, `ps_light`, and `ps_sound`
- explicit scenario rule conditions and preconditions

## 2. Counting policy

These counts still measure **relevant categorical support**, not full Cartesian products over every input field.

Included in the count when they vary meaningfully for the scenario:
- `state_current`
- `precondition`
- `ps_time`
- `ps_dayType`
- compatible `(ps_motion × activityState)`
- `ps_phone`
- `ps_light`
- `ps_sound`
- `networkType`
- `isCharging`
- additional categorical rule features only when they vary

Excluded from the count:
- numeric / continuous fields such as `state_duration_sec`, `timestep`, `batteryLevel`, `activityDuration`, `cal_eventCount`
- `hour` because it is derived from `ps_time`
- user-profile features
- redundant derived fields like `ps_location` and `transportMode` when already implied by `state_current` and `ps_motion`

## 3. Top-level result

- Total scenarios: **65**
- Strict validated total unique samples across all scenarios: **237,178**
- Earlier overestimated scenarios corrected: **14**

## 4. Summary table

| scenarioId | strict validated unique sample count |
| --- | ---: |
| `AFTER_LUNCH_WALK` | 2560 |
| `ARRIVE_DINING` | 16128 |
| `ARRIVE_OFFICE` | 25920 |
| `ARRIVE_TRANSIT_HUB` | 6912 |
| `CAFE_QUIET` | 768 |
| `CAFE_STAY` | 4608 |
| `CAL_HAS_EVENTS` | 3648 |
| `COMMUTE_EVENING` | 5888 |
| `COMMUTE_FROM_HOME` | 180 |
| `COMMUTE_MORNING` | 2880 |
| `CYCLING` | 864 |
| `DELIVERY_AT_OFFICE` | 2432 |
| `EDUCATION_LONG_SIT` | 3072 |
| `EDUCATION_WAITING` | 5760 |
| `EVENING_AT_OFFICE` | 1920 |
| `FLIGHT_BOARDING` | 1836 |
| `GYM_LONG_STAY` | 1296 |
| `GYM_REST` | 432 |
| `GYM_WORKOUT` | 216 |
| `HOME_AFTERNOON` | 5760 |
| `HOME_AFTER_GYM` | 6336 |
| `HOME_DARK_LONG` | 2160 |
| `HOME_EVENING` | 8832 |
| `HOME_EVENING_DARK` | 2304 |
| `HOME_EVENING_NOISY` | 864 |
| `HOME_LATE_NIGHT` | 1536 |
| `HOME_LUNCH` | 960 |
| `HOME_TO_GYM` | 3240 |
| `HOSPITAL_APPOINTMENT` | 2052 |
| `HOTEL_CHECKIN` | 4608 |
| `IN_MEETING` | 1472 |
| `LATE_NIGHT_NOISY` | 756 |
| `LATE_NIGHT_OVERTIME` | 3840 |
| `LATE_NIGHT_PHONE` | 288 |
| `LATE_RETURN_HOME` | 1152 |
| `LEAVE_OFFICE` | 648 |
| `LONG_DRIVE` | 288 |
| `LONG_OUTDOOR_WALK` | 162 |
| `MEETING_UPCOMING` | 4864 |
| `MORNING_COFFEE` | 3456 |
| `MORNING_EXERCISE` | 648 |
| `MOVIE_TICKET` | 14976 |
| `NOISY_GATHERING` | 1512 |
| `NO_MEETINGS` | 2880 |
| `OFFICE_AFTERNOON` | 2208 |
| `OFFICE_FOCUS_LONG` | 192 |
| `OFFICE_LONG_SESSION` | 2208 |
| `OFFICE_LUNCH_OUT` | 1752 |
| `OFFICE_TO_CAFE` | 9216 |
| `OUTDOOR_RUNNING` | 486 |
| `PARK_WALK` | 486 |
| `QUIET_HOME` | 7920 |
| `REMOTE_MEETING` | 24576 |
| `RETURN_OFFICE_AFTER_LUNCH` | 2304 |
| `RIDESHARE_PICKUP` | 4320 |
| `SCHOOL_QUIET` | 768 |
| `SHOPPING` | 2592 |
| `TRAIN_DEPARTURE` | 3348 |
| `UNKNOWN_LONG_STAY` | 2160 |
| `WAKE_UP` | 576 |
| `WEEKDAY_HOME_DAY` | 960 |
| `WEEKEND_MORNING` | 896 |
| `WEEKEND_OUTDOOR_WALK` | 216 |
| `WEEKEND_OUTING` | 3240 |
| `WEEKEND_OVERTIME` | 3840 |

## 5. Scenarios corrected from the earlier approximate report

| scenarioId | earlier reported | strict validated |
| --- | ---: | ---: |
| `AFTER_LUNCH_WALK` | 7680 | 2560 |
| `HOME_LATE_NIGHT` | 3072 | 1536 |
| `HOME_LUNCH` | 2880 | 960 |
| `IN_MEETING` | 2944 | 1472 |
| `LATE_NIGHT_NOISY` | 4212 | 756 |
| `LATE_NIGHT_PHONE` | 576 | 288 |
| `LATE_RETURN_HOME` | 4608 | 1152 |
| `MORNING_COFFEE` | 6912 | 3456 |
| `MORNING_EXERCISE` | 1944 | 648 |
| `NOISY_GATHERING` | 4212 | 1512 |
| `QUIET_HOME` | 15840 | 7920 |
| `WAKE_UP` | 1152 | 576 |
| `WEEKEND_OUTDOOR_WALK` | 324 | 216 |
| `WEEKEND_OUTING` | 4860 | 3240 |

## 6. Notes

- A lower strict count does **not** mean the scenario is weak; it means some earlier combinations were not actually valid once all state-dependent and rule-dependent constraints were enforced together.
- These counts are the correct basis for deciding whether to keep all unique supports or augment to a minimum row target such as 2,000.


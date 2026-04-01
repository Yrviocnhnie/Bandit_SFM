# Scenario Unique Samples Count Report

## 1. Counting policy used for unique-sample support

This report counts only the **relevant categorical variability** for each scenario.

### Included in the unique-sample count
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
- any additional categorical rule feature only when it varies

### Explicitly excluded from the unique-sample count
- numeric / continuous fields: `state_duration_sec`, `timestep`, `batteryLevel`, `activityDuration`, `cal_eventCount`
- `hour` because it is derived from `ps_time`
- user-profile features: `user_id_hash_bucket`, `age_bucket`, `sex`, `has_kids`
- redundant derived fields like `ps_location` and `transportMode` when they are already implied by `state_current` and `ps_motion`

## 2. Summary table: unique-sample counts by scenario

| scenarioId | unique sample count |
| --- | ---: |
| `ARRIVE_OFFICE` | 25920 |
| `REMOTE_MEETING` | 24576 |
| `ARRIVE_DINING` | 16128 |
| `QUIET_HOME` | 15840 |
| `MOVIE_TICKET` | 14976 |
| `OFFICE_TO_CAFE` | 9216 |
| `HOME_EVENING` | 8832 |
| `AFTER_LUNCH_WALK` | 7680 |
| `ARRIVE_TRANSIT_HUB` | 6912 |
| `MORNING_COFFEE` | 6912 |
| `HOME_AFTER_GYM` | 6336 |
| `COMMUTE_EVENING` | 5888 |
| `EDUCATION_WAITING` | 5760 |
| `HOME_AFTERNOON` | 5760 |
| `MEETING_UPCOMING` | 4864 |
| `WEEKEND_OUTING` | 4860 |
| `CAFE_STAY` | 4608 |
| `HOTEL_CHECKIN` | 4608 |
| `LATE_RETURN_HOME` | 4608 |
| `RIDESHARE_PICKUP` | 4320 |
| `LATE_NIGHT_NOISY` | 4212 |
| `NOISY_GATHERING` | 4212 |
| `LATE_NIGHT_OVERTIME` | 3840 |
| `WEEKEND_OVERTIME` | 3840 |
| `CAL_HAS_EVENTS` | 3648 |
| `TRAIN_DEPARTURE` | 3348 |
| `HOME_TO_GYM` | 3240 |
| `EDUCATION_LONG_SIT` | 3072 |
| `HOME_LATE_NIGHT` | 3072 |
| `IN_MEETING` | 2944 |
| `COMMUTE_MORNING` | 2880 |
| `HOME_LUNCH` | 2880 |
| `NO_MEETINGS` | 2880 |
| `SHOPPING` | 2592 |
| `DELIVERY_AT_OFFICE` | 2432 |
| `HOME_EVENING_DARK` | 2304 |
| `RETURN_OFFICE_AFTER_LUNCH` | 2304 |
| `OFFICE_AFTERNOON` | 2208 |
| `OFFICE_LONG_SESSION` | 2208 |
| `HOME_DARK_LONG` | 2160 |
| `UNKNOWN_LONG_STAY` | 2160 |
| `HOSPITAL_APPOINTMENT` | 2052 |
| `MORNING_EXERCISE` | 1944 |
| `EVENING_AT_OFFICE` | 1920 |
| `FLIGHT_BOARDING` | 1836 |
| `OFFICE_LUNCH_OUT` | 1752 |
| `GYM_LONG_STAY` | 1296 |
| `WAKE_UP` | 1152 |
| `WEEKDAY_HOME_DAY` | 960 |
| `WEEKEND_MORNING` | 896 |
| `CYCLING` | 864 |
| `HOME_EVENING_NOISY` | 864 |
| `CAFE_QUIET` | 768 |
| `SCHOOL_QUIET` | 768 |
| `LEAVE_OFFICE` | 648 |
| `LATE_NIGHT_PHONE` | 576 |
| `OUTDOOR_RUNNING` | 486 |
| `PARK_WALK` | 486 |
| `GYM_REST` | 432 |
| `WEEKEND_OUTDOOR_WALK` | 324 |
| `LONG_DRIVE` | 288 |
| `GYM_WORKOUT` | 216 |
| `OFFICE_FOCUS_LONG` | 192 |
| `COMMUTE_FROM_HOME` | 180 |
| `LONG_OUTDOOR_WALK` | 162 |

## 3. Scenario-wise breakdown

### ARRIVE_OFFICE

- **Unique-sample count:** **25920**
- **Allowed `state_current` values:** `office_arriving`, `office_working`, `office_lunch_break`, `office_overtime`, `office_late_overtime`, `office_rest_day`
- **Allowed `precondition` values:** `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_arriving` | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | `dawn`, `morning` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1920 |
| `office_working` | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1920 |
| `office_lunch_break` | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 960 |
| `office_overtime` | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | `evening`, `night` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1920 |
| `office_late_overtime` | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | `late_night`, `sleeping` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1920 |
| `office_rest_day` | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 17280 |

Formula: 1920 + 1920 + 960 + 1920 + 1920 + 17280 = **25920**

### OFFICE_LUNCH_OUT

- **Unique-sample count:** **1752**
- **Allowed `state_current` values:** `outdoor_walking`, `at_restaurant_lunch`, `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition` values:** `office_arriving`, `office_working`, `office_lunch_break`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `1`; `wifiLostCategory` = `work`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `outdoor_walking` | `office_arriving`, `office_working`, `office_lunch_break` | `forenoon`, `lunch` | `workday` | walking: active, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal`, `quiet` | `cellular` | `0` | 216 |
| `at_restaurant_lunch` | `office_arriving`, `office_working`, `office_lunch_break` | `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 384 |
| `at_cafe` | `office_arriving`, `office_working`, `office_lunch_break` | `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 768 |
| `at_cafe_quiet` | `office_arriving`, `office_working`, `office_lunch_break` | `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 384 |

Formula: 216 + 384 + 768 + 384 = **1752**

### OFFICE_AFTERNOON

- **Unique-sample count:** **2208**
- **Allowed `state_current` values:** `office_working`, `office_working_focused`, `office_working_noisy`
- **Allowed `precondition` values:** `office_arriving`, `office_working`, `office_lunch_break`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_working` | `office_arriving`, `office_working`, `office_lunch_break` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1440 |
| `office_working_focused` | `office_arriving`, `office_working`, `office_lunch_break` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 288 |
| `office_working_noisy` | `office_arriving`, `office_working`, `office_lunch_break` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy` | `cellular`, `wifi` | `0` | 480 |

Formula: 1440 + 288 + 480 = **2208**

### LEAVE_OFFICE

- **Unique-sample count:** **648**
- **Allowed `state_current` values:** `commuting_walk_home`, `commuting_cycle_home`, `commuting_drive_home`, `commuting_transit_home`, `outdoor_walking`
- **Allowed `precondition` values:** `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `1`; `wifiLostCategory` = `work`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `commuting_walk_home` | `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy` | `evening`, `night` | `workday` | walking: active, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 144 |
| `commuting_cycle_home` | `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy` | `evening`, `night` | `workday` | cycling: active | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 72 |
| `commuting_drive_home` | `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy` | `evening`, `night` | `workday` | driving: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 144 |
| `commuting_transit_home` | `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy` | `evening`, `night` | `workday` | transit: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 144 |
| `outdoor_walking` | `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy` | `evening`, `night` | `workday` | walking: active, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 144 |

Formula: 144 + 72 + 144 + 144 + 144 = **648**

### WEEKDAY_HOME_DAY

- **Unique-sample count:** **960**
- **Allowed `state_current` values:** `home_daytime_workday`, `home_daytime_workday_dark`, `home_daytime_workday_lying`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_workday_lying`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_daytime_workday` | `home_morning_workday`, `home_morning_workday_lying` | `afternoon`, `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 576 |
| `home_daytime_workday_dark` | `home_morning_workday`, `home_morning_workday_lying` | `afternoon`, `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dark` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 288 |
| `home_daytime_workday_lying` | `home_morning_workday`, `home_morning_workday_lying` | `afternoon`, `forenoon`, `lunch` | `workday` | walking: sitting; stationary: sitting | `holding_lying` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 96 |

Formula: 576 + 288 + 96 = **960**

### ARRIVE_TRANSIT_HUB

- **Unique-sample count:** **6912**
- **Allowed `state_current` values:** `at_metro`, `at_rail_station`, `at_airport`, `at_transit_hub`
- **Allowed `precondition` values:** `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_metro` | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 1728 |
| `at_rail_station` | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 1728 |
| `at_airport` | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 1728 |
| `at_transit_hub` | `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 1728 |

Formula: 1728 + 1728 + 1728 + 1728 = **6912**

### LATE_NIGHT_OVERTIME

- **Unique-sample count:** **3840**
- **Allowed `state_current` values:** `office_overtime`, `office_late_overtime`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `office_working_noisy`, `office_overtime`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_overtime` | `office_working`, `office_working_focused`, `office_working_noisy`, `office_overtime` | `evening`, `night` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1920 |
| `office_late_overtime` | `office_working`, `office_working_focused`, `office_working_noisy`, `office_overtime` | `late_night`, `sleeping` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1920 |

Formula: 1920 + 1920 = **3840**

### WEEKEND_OVERTIME

- **Unique-sample count:** **3840**
- **Allowed `state_current` values:** `office_rest_day`
- **Allowed `precondition` values:** `office_arriving`, `office_rest_day`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_rest_day` | `office_arriving`, `office_rest_day` | `afternoon`, `forenoon`, `lunch`, `morning` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 3840 |

Formula: 3840 = **3840**

### EVENING_AT_OFFICE

- **Unique-sample count:** **1920**
- **Allowed `state_current` values:** `office_overtime`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `office_working_noisy`, `office_overtime`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_overtime` | `office_working`, `office_working_focused`, `office_working_noisy`, `office_overtime` | `evening`, `night` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1920 |

Formula: 1920 = **1920**

### WAKE_UP

- **Unique-sample count:** **1152**
- **Allowed `state_current` values:** `home_morning_workday`, `home_morning_rest`
- **Allowed `precondition` values:** `home_sleeping`, `home_sleeping_lying`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_morning_workday` | `home_sleeping`, `home_sleeping_lying` | `dawn`, `morning` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 384 |
| `home_morning_rest` | `home_sleeping`, `home_sleeping_lying` | `dawn`, `morning` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 768 |

Formula: 384 + 768 = **1152**

### MORNING_EXERCISE

- **Unique-sample count:** **1944**
- **Allowed `state_current` values:** `at_gym`, `at_gym_exercising`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_rest`, `home_morning_rest_lying`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_gym` | `home_morning_workday`, `home_morning_rest`, `home_morning_rest_lying` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `normal` | `noisy`, `normal` | `cellular` | `0` | 1296 |
| `at_gym_exercising` | `home_morning_workday`, `home_morning_rest`, `home_morning_rest_lying` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | cycling: active; running: active | `face_up`, `in_pocket`, `in_use` | `bright`, `normal` | `noisy`, `normal` | `cellular` | `0` | 648 |

Formula: 1296 + 648 = **1944**

### SHOPPING

- **Unique-sample count:** **2592**
- **Allowed `state_current` values:** `at_shopping`
- **Allowed `precondition` values:** `commuting_drive_out`, `commuting_walk_out`, `outdoor_walking`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_shopping` | `commuting_drive_out`, `commuting_walk_out`, `outdoor_walking` | `afternoon`, `evening`, `forenoon`, `lunch`, `morning`, `night` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 2592 |

Formula: 2592 = **2592**

### HOME_EVENING

- **Unique-sample count:** **8832**
- **Allowed `state_current` values:** `home_evening`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy`
- **Allowed `precondition` values:** `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_evening` | `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home` | `evening`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 4608 |
| `home_evening_dark` | `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home` | `evening`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dark` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 2304 |
| `home_evening_lying` | `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home` | `evening`, `night` | `holiday`, `weekend`, `workday` | walking: sitting; stationary: sitting | `holding_lying` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 768 |
| `home_evening_noisy` | `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home` | `evening`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `noisy` | `cellular`, `wifi` | `0` | 1152 |

Formula: 4608 + 2304 + 768 + 1152 = **8832**

### ARRIVE_DINING

- **Unique-sample count:** **16128**
- **Allowed `state_current` values:** `at_restaurant_lunch`, `at_restaurant_dinner`, `at_restaurant_other`, `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition` values:** `outdoor_walking`, `at_shopping`, `commuting_walk_out`, `commuting_drive_out`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_restaurant_lunch` | `outdoor_walking`, `at_shopping`, `commuting_walk_out`, `commuting_drive_out` | `lunch` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 1536 |
| `at_restaurant_dinner` | `outdoor_walking`, `at_shopping`, `commuting_walk_out`, `commuting_drive_out` | `evening`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 3072 |
| `at_restaurant_other` | `outdoor_walking`, `at_shopping`, `commuting_walk_out`, `commuting_drive_out` | `afternoon` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 2304 |
| `at_cafe` | `outdoor_walking`, `at_shopping`, `commuting_walk_out`, `commuting_drive_out` | `afternoon`, `evening`, `lunch`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 6144 |
| `at_cafe_quiet` | `outdoor_walking`, `at_shopping`, `commuting_walk_out`, `commuting_drive_out` | `afternoon`, `evening`, `lunch`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 3072 |

Formula: 1536 + 3072 + 2304 + 6144 + 3072 = **16128**

### HOME_LATE_NIGHT

- **Unique-sample count:** **3072**
- **Allowed `state_current` values:** `home_sleeping`, `home_sleeping_lying`
- **Allowed `precondition` values:** `home_evening`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_sleeping` | `home_evening`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy` | `late_night`, `sleeping` | `holiday`, `weekend`, `workday` | walking: sleeping; stationary: sleeping | `face_up`, `in_use`, `on_desk` | `dark`, `dim` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 2304 |
| `home_sleeping_lying` | `home_evening`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy` | `late_night`, `sleeping` | `holiday`, `weekend`, `workday` | walking: sleeping; stationary: sleeping | `holding_lying` | `dark`, `dim` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 768 |

Formula: 2304 + 768 = **3072**

### WEEKEND_MORNING

- **Unique-sample count:** **896**
- **Allowed `state_current` values:** `home_morning_rest`, `home_morning_rest_lying`
- **Allowed `precondition` values:** `home_sleeping`, `home_sleeping_lying`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_morning_rest` | `home_sleeping`, `home_sleeping_lying` | `dawn`, `morning` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 768 |
| `home_morning_rest_lying` | `home_sleeping`, `home_sleeping_lying` | `dawn`, `morning` | `holiday`, `weekend` | walking: sitting; stationary: sitting | `holding_lying` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 128 |

Formula: 768 + 128 = **896**

### WEEKEND_OUTING

- **Unique-sample count:** **4860**
- **Allowed `state_current` values:** `outdoor_walking`, `at_shopping`
- **Allowed `precondition` values:** `home_morning_rest`, `home_morning_rest_lying`, `home_daytime_rest`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `outdoor_walking` | `home_morning_rest`, `home_morning_rest_lying`, `home_daytime_rest` | `afternoon`, `dawn`, `evening`, `forenoon`, `lunch`, `morning`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal`, `quiet` | `cellular` | `0` | 2268 |
| `at_shopping` | `home_morning_rest`, `home_morning_rest_lying`, `home_daytime_rest` | `afternoon`, `evening`, `forenoon`, `lunch`, `morning`, `night` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 2592 |

Formula: 2268 + 2592 = **4860**

### PARK_WALK

- **Unique-sample count:** **486**
- **Allowed `state_current` values:** `outdoor_walking`
- **Allowed `precondition` values:** `home_daytime_rest`, `home_daytime_workday`, `outdoor_walking`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `outdoor_walking` | `home_daytime_rest`, `home_daytime_workday`, `outdoor_walking` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | walking: active, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 486 |

Formula: 486 = **486**

### LONG_DRIVE

- **Unique-sample count:** **288**
- **Allowed `state_current` values:** `driving`, `commuting_drive_out`, `commuting_drive_home`
- **Allowed `precondition` values:** `commuting_drive_out`, `driving`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `driving` | `commuting_drive_out`, `driving` | `dawn`, `evening`, `morning`, `night` | `workday` | driving: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 144 |
| `commuting_drive_out` | `commuting_drive_out`, `driving` | `dawn`, `morning` | `workday` | driving: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 72 |
| `commuting_drive_home` | `commuting_drive_out`, `driving` | `evening`, `night` | `workday` | driving: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 72 |

Formula: 144 + 72 + 72 = **288**

### LATE_RETURN_HOME

- **Unique-sample count:** **4608**
- **Allowed `state_current` values:** `home_sleeping`, `home_sleeping_lying`
- **Allowed `precondition` values:** `office_overtime`, `office_late_overtime`, `driving`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_sleeping` | `office_overtime`, `office_late_overtime`, `driving`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home` | `late_night`, `sleeping` | `holiday`, `weekend`, `workday` | walking: sleeping; stationary: sleeping | `face_up`, `in_use`, `on_desk` | `dark`, `dim` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 3456 |
| `home_sleeping_lying` | `office_overtime`, `office_late_overtime`, `driving`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home` | `late_night`, `sleeping` | `holiday`, `weekend`, `workday` | walking: sleeping; stationary: sleeping | `holding_lying` | `dark`, `dim` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 1152 |

Formula: 3456 + 1152 = **4608**

### HOME_AFTERNOON

- **Unique-sample count:** **5760**
- **Allowed `state_current` values:** `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying`
- **Allowed `precondition` values:** `home_morning_rest`, `home_morning_workday`, `home_daytime_workday`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_daytime_rest` | `home_morning_rest`, `home_morning_workday`, `home_daytime_workday` | `afternoon`, `forenoon`, `lunch` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 3456 |
| `home_daytime_rest_dark` | `home_morning_rest`, `home_morning_workday`, `home_daytime_workday` | `afternoon`, `forenoon`, `lunch` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dark` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 1728 |
| `home_daytime_rest_lying` | `home_morning_rest`, `home_morning_workday`, `home_daytime_workday` | `afternoon`, `forenoon`, `lunch` | `holiday`, `weekend` | walking: sitting; stationary: sitting | `holding_lying` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 576 |

Formula: 3456 + 1728 + 576 = **5760**

### MORNING_COFFEE

- **Unique-sample count:** **6912**
- **Allowed `state_current` values:** `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_rest`, `commuting_walk_out`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_cafe` | `home_morning_workday`, `home_morning_rest`, `commuting_walk_out` | `afternoon`, `evening`, `lunch`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 4608 |
| `at_cafe_quiet` | `home_morning_workday`, `home_morning_rest`, `commuting_walk_out` | `afternoon`, `evening`, `lunch`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 2304 |

Formula: 4608 + 2304 = **6912**

### CAFE_STAY

- **Unique-sample count:** **4608**
- **Allowed `state_current` values:** `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition` values:** `at_cafe`, `at_cafe_quiet`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_cafe` | `at_cafe`, `at_cafe_quiet` | `afternoon`, `evening`, `lunch`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 3072 |
| `at_cafe_quiet` | `at_cafe`, `at_cafe_quiet` | `afternoon`, `evening`, `lunch`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 1536 |

Formula: 3072 + 1536 = **4608**

### QUIET_HOME

- **Unique-sample count:** **15840**
- **Allowed `state_current` values:** `home_evening`, `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`
- **Allowed `precondition` values:** `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`, `home_evening`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_evening` | `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`, `home_evening` | `evening`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 5760 |
| `home_daytime_workday` | `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`, `home_evening` | `afternoon`, `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 1440 |
| `home_daytime_rest` | `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`, `home_evening` | `afternoon`, `forenoon`, `lunch` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 5760 |
| `home_morning_workday` | `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`, `home_evening` | `dawn`, `morning` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 960 |
| `home_morning_rest` | `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`, `home_evening` | `dawn`, `morning` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 1920 |

Formula: 5760 + 1440 + 5760 + 960 + 1920 = **15840**

### GYM_WORKOUT

- **Unique-sample count:** **216**
- **Allowed `state_current` values:** `at_gym_exercising`
- **Allowed `precondition` values:** `at_gym`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_gym_exercising` | `at_gym` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | cycling: active; running: active | `face_up`, `in_pocket`, `in_use` | `bright`, `normal` | `noisy`, `normal` | `cellular` | `0` | 216 |

Formula: 216 = **216**

### GYM_REST

- **Unique-sample count:** **432**
- **Allowed `state_current` values:** `at_gym`
- **Allowed `precondition` values:** `at_gym_exercising`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_gym` | `at_gym_exercising` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `normal` | `noisy`, `normal` | `cellular` | `0` | 432 |

Formula: 432 = **432**

### GYM_LONG_STAY

- **Unique-sample count:** **1296**
- **Allowed `state_current` values:** `at_gym`, `at_gym_exercising`
- **Allowed `precondition` values:** `at_gym`, `at_gym_exercising`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_gym` | `at_gym`, `at_gym_exercising` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `normal` | `noisy`, `normal` | `cellular` | `0` | 864 |
| `at_gym_exercising` | `at_gym`, `at_gym_exercising` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | cycling: active; running: active | `face_up`, `in_pocket`, `in_use` | `bright`, `normal` | `noisy`, `normal` | `cellular` | `0` | 432 |

Formula: 864 + 432 = **1296**

### OUTDOOR_RUNNING

- **Unique-sample count:** **486**
- **Allowed `state_current` values:** `outdoor_running`
- **Allowed `precondition` values:** `home_morning_rest`, `home_morning_workday`, `outdoor_walking`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `outdoor_running` | `home_morning_rest`, `home_morning_workday`, `outdoor_walking` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | running: active | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal`, `quiet` | `cellular` | `0` | 486 |

Formula: 486 = **486**

### NOISY_GATHERING

- **Unique-sample count:** **4212**
- **Allowed `state_current` values:** `unknown_noisy`, `at_social`
- **Allowed `precondition` values:** `home_evening`, `at_restaurant_dinner`, `at_social`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `unknown_noisy` | `home_evening`, `at_restaurant_dinner`, `at_social` | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy` | `cellular`, `wifi` | `0` | 2916 |
| `at_social` | `home_evening`, `at_restaurant_dinner`, `at_social` | `evening`, `late_night`, `night` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `dim`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 1296 |

Formula: 2916 + 1296 = **4212**

### WEEKEND_OUTDOOR_WALK

- **Unique-sample count:** **324**
- **Allowed `state_current` values:** `outdoor_walking`
- **Allowed `precondition` values:** `home_morning_rest`, `home_daytime_rest`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `outdoor_walking` | `home_morning_rest`, `home_daytime_rest` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | walking: active, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 324 |

Formula: 324 = **324**

### LONG_OUTDOOR_WALK

- **Unique-sample count:** **162**
- **Allowed `state_current` values:** `outdoor_walking`
- **Allowed `precondition` values:** `outdoor_walking`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `outdoor_walking` | `outdoor_walking` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | walking: active, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 162 |

Formula: 162 = **162**

### CYCLING

- **Unique-sample count:** **864**
- **Allowed `state_current` values:** `outdoor_cycling`, `commuting_cycle_out`, `commuting_cycle_home`
- **Allowed `precondition` values:** `commuting_cycle_out`, `commuting_cycle_home`, `outdoor_cycling`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `outdoor_cycling` | `commuting_cycle_out`, `commuting_cycle_home`, `outdoor_cycling` | `afternoon`, `dawn`, `evening`, `morning` | `holiday`, `weekend`, `workday` | cycling: active | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal`, `quiet` | `cellular` | `0` | 648 |
| `commuting_cycle_out` | `commuting_cycle_out`, `commuting_cycle_home`, `outdoor_cycling` | `dawn`, `morning` | `workday` | cycling: active | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal`, `quiet` | `cellular` | `0` | 108 |
| `commuting_cycle_home` | `commuting_cycle_out`, `commuting_cycle_home`, `outdoor_cycling` | `evening`, `night` | `workday` | cycling: active | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal`, `quiet` | `cellular` | `0` | 108 |

Formula: 648 + 108 + 108 = **864**

### LATE_NIGHT_NOISY

- **Unique-sample count:** **4212**
- **Allowed `state_current` values:** `unknown_noisy`, `at_social`
- **Allowed `precondition` values:** `home_evening`, `at_social`, `at_restaurant_dinner`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `unknown_noisy` | `home_evening`, `at_social`, `at_restaurant_dinner` | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy` | `cellular`, `wifi` | `0` | 2916 |
| `at_social` | `home_evening`, `at_social`, `at_restaurant_dinner` | `evening`, `late_night`, `night` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `dim`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 1296 |

Formula: 2916 + 1296 = **4212**

### EDUCATION_WAITING

- **Unique-sample count:** **5760**
- **Allowed `state_current` values:** `at_education`, `at_education_class`, `at_education_break`
- **Allowed `precondition` values:** `commuting_walk_out`, `commuting_transit_out`, `at_education`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_education` | `commuting_walk_out`, `commuting_transit_out`, `at_education` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 3456 |
| `at_education_class` | `commuting_walk_out`, `commuting_transit_out`, `at_education` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 1152 |
| `at_education_break` | `commuting_walk_out`, `commuting_transit_out`, `at_education` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy` | `cellular`, `wifi` | `0` | 1152 |

Formula: 3456 + 1152 + 1152 = **5760**

### EDUCATION_LONG_SIT

- **Unique-sample count:** **3072**
- **Allowed `state_current` values:** `at_education`, `at_education_class`
- **Allowed `precondition` values:** `at_education`, `at_education_class`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_education` | `at_education`, `at_education_class` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 2304 |
| `at_education_class` | `at_education`, `at_education_class` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 768 |

Formula: 2304 + 768 = **3072**

### HOME_LUNCH

- **Unique-sample count:** **2880**
- **Allowed `state_current` values:** `home_daytime_workday`, `home_daytime_rest`
- **Allowed `precondition` values:** `home_daytime_workday`, `home_daytime_rest`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_daytime_workday` | `home_daytime_workday`, `home_daytime_rest` | `afternoon`, `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 576 |
| `home_daytime_rest` | `home_daytime_workday`, `home_daytime_rest` | `afternoon`, `forenoon`, `lunch` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 2304 |

Formula: 576 + 2304 = **2880**

### AFTER_LUNCH_WALK

- **Unique-sample count:** **7680**
- **Allowed `state_current` values:** `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying`
- **Allowed `precondition` values:** `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying`, `home_daytime_workday`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_daytime_rest` | `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying`, `home_daytime_workday` | `afternoon`, `forenoon`, `lunch` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 4608 |
| `home_daytime_rest_dark` | `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying`, `home_daytime_workday` | `afternoon`, `forenoon`, `lunch` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dark` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 2304 |
| `home_daytime_rest_lying` | `home_daytime_rest`, `home_daytime_rest_dark`, `home_daytime_rest_lying`, `home_daytime_workday` | `afternoon`, `forenoon`, `lunch` | `holiday`, `weekend` | walking: sitting; stationary: sitting | `holding_lying` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 768 |

Formula: 4608 + 2304 + 768 = **7680**

### COMMUTE_MORNING

- **Unique-sample count:** **2880**
- **Allowed `state_current` values:** `office_arriving`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_workday_lying`, `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_arriving` | `home_morning_workday`, `home_morning_workday_lying`, `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out` | `dawn`, `morning` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 2880 |

Formula: 2880 = **2880**

### COMMUTE_EVENING

- **Unique-sample count:** **5888**
- **Allowed `state_current` values:** `home_evening`, `home_evening_dark`, `home_evening_lying`, `home_evening_noisy`
- **Allowed `precondition` values:** `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_evening` | `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home` | `evening`, `night` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 3072 |
| `home_evening_dark` | `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home` | `evening`, `night` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dark` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 1536 |
| `home_evening_lying` | `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home` | `evening`, `night` | `workday` | walking: sitting; stationary: sitting | `holding_lying` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 512 |
| `home_evening_noisy` | `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy`, `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `commuting_cycle_home` | `evening`, `night` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `noisy` | `cellular`, `wifi` | `0` | 768 |

Formula: 3072 + 1536 + 512 + 768 = **5888**

### HOME_TO_GYM

- **Unique-sample count:** **3240**
- **Allowed `state_current` values:** `at_gym`, `at_gym_exercising`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_rest`, `home_daytime_workday`, `home_daytime_rest`, `home_evening`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_gym` | `home_morning_workday`, `home_morning_rest`, `home_daytime_workday`, `home_daytime_rest`, `home_evening` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `normal` | `noisy`, `normal` | `cellular` | `0` | 2160 |
| `at_gym_exercising` | `home_morning_workday`, `home_morning_rest`, `home_daytime_workday`, `home_daytime_rest`, `home_evening` | `afternoon`, `evening`, `morning` | `holiday`, `weekend`, `workday` | cycling: active; running: active | `face_up`, `in_pocket`, `in_use` | `bright`, `normal` | `noisy`, `normal` | `cellular` | `0` | 1080 |

Formula: 2160 + 1080 = **3240**

### DELIVERY_AT_OFFICE

- **Unique-sample count:** **2432**
- **Allowed `state_current` values:** `office_arriving`, `office_working`, `office_working_focused`, `office_working_noisy`
- **Allowed `precondition` values:** `office_working`, `office_arriving`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `1`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_arriving` | `office_working`, `office_arriving` | `dawn`, `morning` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 960 |
| `office_working` | `office_working`, `office_arriving` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 960 |
| `office_working_focused` | `office_working`, `office_arriving` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 192 |
| `office_working_noisy` | `office_working`, `office_arriving` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy` | `cellular`, `wifi` | `0` | 320 |

Formula: 960 + 960 + 192 + 320 = **2432**

### MEETING_UPCOMING

- **Unique-sample count:** **4864**
- **Allowed `state_current` values:** `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_working` | `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1920 |
| `office_working_focused` | `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 384 |
| `home_daytime_workday` | `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 1536 |
| `at_cafe_quiet` | `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon`, `lunch`, `morning` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 1024 |

Formula: 1920 + 384 + 1536 + 1024 = **4864**

### IN_MEETING

- **Unique-sample count:** **2944**
- **Allowed `state_current` values:** `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `1`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_working_focused` | `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 384 |
| `home_daytime_workday` | `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 1536 |
| `at_cafe_quiet` | `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon`, `lunch`, `morning` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 1024 |

Formula: 384 + 1536 + 1024 = **2944**

### REMOTE_MEETING

- **Unique-sample count:** **24576**
- **Allowed `state_current` values:** `home_daytime_workday`, `at_cafe_quiet`
- **Allowed `precondition` values:** `home_daytime_workday`, `at_cafe_quiet`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`
- **Additional varying categorical rule features included in the count:** `cal_nextLocation` in (`airport`, `cafe`, `custom`, `education`, `en_route`, `gym`, `health`, `home`, `metro`, `outdoor`, `rail_station`, `restaurant`, `shopping`, `social`, `transit`, `work`)

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_daytime_workday` | `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 12288 |
| `at_cafe_quiet` | `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `evening`, `forenoon`, `lunch`, `morning`, `night` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 12288 |

Formula: 12288 + 12288 = **24576**

### NO_MEETINGS

- **Unique-sample count:** **2880**
- **Allowed `state_current` values:** `office_arriving`, `office_working`
- **Allowed `precondition` values:** `office_arriving`, `office_working`, `home_daytime_workday`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_arriving` | `office_arriving`, `office_working`, `home_daytime_workday` | `dawn`, `morning` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1440 |
| `office_working` | `office_arriving`, `office_working`, `home_daytime_workday` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1440 |

Formula: 1440 + 1440 = **2880**

### CAL_HAS_EVENTS

- **Unique-sample count:** **3648**
- **Allowed `state_current` values:** `office_working`, `office_working_focused`, `home_daytime_workday`, `at_cafe_quiet`
- **Allowed `precondition` values:** `office_working`, `home_daytime_workday`, `at_cafe_quiet`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_working` | `office_working`, `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1440 |
| `office_working_focused` | `office_working`, `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 288 |
| `home_daytime_workday` | `office_working`, `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 1152 |
| `at_cafe_quiet` | `office_working`, `home_daytime_workday`, `at_cafe_quiet` | `afternoon`, `forenoon`, `lunch`, `morning` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 768 |

Formula: 1440 + 288 + 1152 + 768 = **3648**

### RETURN_OFFICE_AFTER_LUNCH

- **Unique-sample count:** **2304**
- **Allowed `state_current` values:** `office_working`, `office_working_focused`
- **Allowed `precondition` values:** `at_restaurant_lunch`, `at_restaurant_other`, `at_cafe`, `at_cafe_quiet`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_working` | `at_restaurant_lunch`, `at_restaurant_other`, `at_cafe`, `at_cafe_quiet` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1920 |
| `office_working_focused` | `at_restaurant_lunch`, `at_restaurant_other`, `at_cafe`, `at_cafe_quiet` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 384 |

Formula: 1920 + 384 = **2304**

### HOME_AFTER_GYM

- **Unique-sample count:** **6336**
- **Allowed `state_current` values:** `home_evening`, `home_daytime_workday`, `home_daytime_rest`, `home_morning_workday`, `home_morning_rest`
- **Allowed `precondition` values:** `at_gym`, `at_gym_exercising`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_evening` | `at_gym`, `at_gym_exercising` | `evening`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 2304 |
| `home_daytime_workday` | `at_gym`, `at_gym_exercising` | `afternoon`, `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 576 |
| `home_daytime_rest` | `at_gym`, `at_gym_exercising` | `afternoon`, `forenoon`, `lunch` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `bright`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 2304 |
| `home_morning_workday` | `at_gym`, `at_gym_exercising` | `dawn`, `morning` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 384 |
| `home_morning_rest` | `at_gym`, `at_gym_exercising` | `dawn`, `morning` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 768 |

Formula: 2304 + 576 + 2304 + 384 + 768 = **6336**

### COMMUTE_FROM_HOME

- **Unique-sample count:** **180**
- **Allowed `state_current` values:** `commuting_walk_out`, `commuting_cycle_out`, `commuting_drive_out`, `commuting_transit_out`
- **Allowed `precondition` values:** `home_morning_workday`, `home_morning_workday_lying`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `commuting_walk_out` | `home_morning_workday`, `home_morning_workday_lying` | `dawn`, `morning` | `workday` | walking: active, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 72 |
| `commuting_cycle_out` | `home_morning_workday`, `home_morning_workday_lying` | `dawn`, `morning` | `workday` | cycling: active | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 36 |
| `commuting_drive_out` | `home_morning_workday`, `home_morning_workday_lying` | `dawn`, `morning` | `workday` | driving: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 36 |
| `commuting_transit_out` | `home_morning_workday`, `home_morning_workday_lying` | `dawn`, `morning` | `workday` | transit: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 36 |

Formula: 72 + 36 + 36 + 36 = **180**

### OFFICE_TO_CAFE

- **Unique-sample count:** **9216**
- **Allowed `state_current` values:** `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `office_working_noisy`, `office_lunch_break`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_cafe` | `office_working`, `office_working_focused`, `office_working_noisy`, `office_lunch_break` | `afternoon`, `evening`, `lunch`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 6144 |
| `at_cafe_quiet` | `office_working`, `office_working_focused`, `office_working_noisy`, `office_lunch_break` | `afternoon`, `evening`, `lunch`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 3072 |

Formula: 6144 + 3072 = **9216**

### LATE_NIGHT_PHONE

- **Unique-sample count:** **576**
- **Allowed `state_current` values:** `home_sleeping_lying`
- **Allowed `precondition` values:** `home_evening`, `home_evening_lying`, `home_sleeping`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_sleeping_lying` | `home_evening`, `home_evening_lying`, `home_sleeping` | `late_night`, `sleeping` | `holiday`, `weekend`, `workday` | walking: sleeping; stationary: sleeping | `holding_lying` | `dark`, `dim` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 576 |

Formula: 576 = **576**

### OFFICE_FOCUS_LONG

- **Unique-sample count:** **192**
- **Allowed `state_current` values:** `office_working_focused`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_working_focused` | `office_working`, `office_working_focused` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 192 |

Formula: 192 = **192**

### HOME_DARK_LONG

- **Unique-sample count:** **2160**
- **Allowed `state_current` values:** `home_daytime_workday_dark`, `home_daytime_rest_dark`
- **Allowed `precondition` values:** `home_daytime_workday_dark`, `home_daytime_rest_dark`, `home_evening_dark`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_daytime_workday_dark` | `home_daytime_workday_dark`, `home_daytime_rest_dark`, `home_evening_dark` | `afternoon`, `forenoon`, `lunch` | `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dark` | `normal`, `quiet` | `cellular`, `wifi` | `0` | 432 |
| `home_daytime_rest_dark` | `home_daytime_workday_dark`, `home_daytime_rest_dark`, `home_evening_dark` | `afternoon`, `forenoon`, `lunch` | `holiday`, `weekend` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dark` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 1728 |

Formula: 432 + 1728 = **2160**

### UNKNOWN_LONG_STAY

- **Unique-sample count:** **2160**
- **Allowed `state_current` values:** `stationary_unknown`, `unknown_noisy`, `unknown_dark`, `unknown_settled`, `unknown_lying`
- **Allowed `precondition` values:** `stationary_unknown`, `unknown_settled`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `stationary_unknown` | `stationary_unknown`, `unknown_settled` | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 648 |
| `unknown_noisy` | `stationary_unknown`, `unknown_settled` | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up` | `bright`, `dim`, `normal` | `noisy` | `cellular` | `0` | 324 |
| `unknown_dark` | `stationary_unknown`, `unknown_settled` | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up` | `dark` | `noisy`, `normal` | `cellular` | `0` | 216 |
| `unknown_settled` | `stationary_unknown`, `unknown_settled` | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `charging` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `1` | 648 |
| `unknown_lying` | `stationary_unknown`, `unknown_settled` | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | `holiday`, `weekend`, `workday` | stationary: sitting | `holding_lying` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 324 |

Formula: 648 + 324 + 216 + 648 + 324 = **2160**

### OFFICE_LONG_SESSION

- **Unique-sample count:** **2208**
- **Allowed `state_current` values:** `office_working`, `office_working_focused`, `office_working_noisy`
- **Allowed `precondition` values:** `office_working`, `office_working_focused`, `office_working_noisy`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `office_working` | `office_working`, `office_working_focused`, `office_working_noisy` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 1440 |
| `office_working_focused` | `office_working`, `office_working_focused`, `office_working_noisy` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down` | `bright`, `normal` | `noisy`, `normal`, `quiet` | `cellular`, `wifi` | `0` | 288 |
| `office_working_noisy` | `office_working`, `office_working_focused`, `office_working_noisy` | `afternoon`, `forenoon` | `workday` | walking: active, standing; stationary: sitting, standing | `face_down`, `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `noisy` | `cellular`, `wifi` | `0` | 480 |

Formula: 1440 + 288 + 480 = **2208**

### HOME_EVENING_DARK

- **Unique-sample count:** **2304**
- **Allowed `state_current` values:** `home_evening_dark`
- **Allowed `precondition` values:** `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `home_evening`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_evening_dark` | `commuting_drive_home`, `commuting_transit_home`, `commuting_walk_home`, `home_evening` | `evening`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dark` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 2304 |

Formula: 2304 = **2304**

### HOME_EVENING_NOISY

- **Unique-sample count:** **864**
- **Allowed `state_current` values:** `home_evening_noisy`
- **Allowed `precondition` values:** `home_evening`, `at_social`, `at_restaurant_dinner`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_evening_noisy` | `home_evening`, `at_social`, `at_restaurant_dinner` | `evening`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_use`, `on_desk` | `dim`, `normal` | `noisy` | `cellular`, `wifi` | `0` | 864 |

Formula: 864 = **864**

### SCHOOL_QUIET

- **Unique-sample count:** **768**
- **Allowed `state_current` values:** `at_education_class`
- **Allowed `precondition` values:** `at_education`, `at_education_class`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_education_class` | `at_education`, `at_education_class` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 768 |

Formula: 768 = **768**

### CAFE_QUIET

- **Unique-sample count:** **768**
- **Allowed `state_current` values:** `at_cafe_quiet`
- **Allowed `precondition` values:** `at_cafe`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `at_cafe_quiet` | `at_cafe` | `afternoon`, `evening`, `lunch`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `quiet` | `cellular`, `wifi` | `0` | 768 |

Formula: 768 = **768**

### TRAIN_DEPARTURE

- **Unique-sample count:** **3348**
- **Allowed `state_current` values:** `commuting_transit_out`, `at_transit_hub`, `at_rail_station`
- **Allowed `precondition` values:** `commuting_transit_out`, `at_transit_hub`, `at_rail_station`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `1`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `commuting_transit_out` | `commuting_transit_out`, `at_transit_hub`, `at_rail_station` | `dawn`, `morning` | `workday` | transit: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 108 |
| `at_transit_hub` | `commuting_transit_out`, `at_transit_hub`, `at_rail_station` | `afternoon`, `dawn`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 1620 |
| `at_rail_station` | `commuting_transit_out`, `at_transit_hub`, `at_rail_station` | `afternoon`, `dawn`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 1620 |

Formula: 108 + 1620 + 1620 = **3348**

### FLIGHT_BOARDING

- **Unique-sample count:** **1836**
- **Allowed `state_current` values:** `commuting_drive_out`, `commuting_transit_out`, `at_airport`
- **Allowed `precondition` values:** `commuting_drive_out`, `commuting_transit_out`, `at_airport`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `1`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `commuting_drive_out` | `commuting_drive_out`, `commuting_transit_out`, `at_airport` | `dawn`, `morning` | `workday` | driving: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 108 |
| `commuting_transit_out` | `commuting_drive_out`, `commuting_transit_out`, `at_airport` | `dawn`, `morning` | `workday` | transit: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 108 |
| `at_airport` | `commuting_drive_out`, `commuting_transit_out`, `at_airport` | `afternoon`, `dawn`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 1620 |

Formula: 108 + 108 + 1620 = **1836**

### HOTEL_CHECKIN

- **Unique-sample count:** **4608**
- **Allowed `state_current` values:** `commuting_drive_out`, `commuting_transit_out`, `at_airport`, `at_transit_hub`
- **Allowed `precondition` values:** `commuting_drive_out`, `commuting_transit_out`, `at_airport`, `at_transit_hub`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `1`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `commuting_drive_out` | `commuting_drive_out`, `commuting_transit_out`, `at_airport`, `at_transit_hub` | `dawn`, `morning` | `workday` | driving: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 144 |
| `commuting_transit_out` | `commuting_drive_out`, `commuting_transit_out`, `at_airport`, `at_transit_hub` | `dawn`, `morning` | `workday` | transit: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 144 |
| `at_airport` | `commuting_drive_out`, `commuting_transit_out`, `at_airport`, `at_transit_hub` | `afternoon`, `dawn`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 2160 |
| `at_transit_hub` | `commuting_drive_out`, `commuting_transit_out`, `at_airport`, `at_transit_hub` | `afternoon`, `dawn`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular` | `0` | 2160 |

Formula: 144 + 144 + 2160 + 2160 = **4608**

### MOVIE_TICKET

- **Unique-sample count:** **14976**
- **Allowed `state_current` values:** `home_evening`, `at_social`, `at_restaurant_other`
- **Allowed `precondition` values:** `home_evening`, `at_social`, `at_restaurant_other`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `1`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `home_evening` | `home_evening`, `at_social`, `at_restaurant_other` | `evening`, `night` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `normal`, `quiet` | `cellular`, `wifi` | `0`, `1` | 4608 |
| `at_social` | `home_evening`, `at_social`, `at_restaurant_other` | `evening`, `late_night`, `night` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `dim`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 1728 |
| `at_restaurant_other` | `home_evening`, `at_social`, `at_restaurant_other` | `afternoon`, `dawn`, `forenoon`, `late_night`, `morning` | `holiday`, `weekend`, `workday` | walking: active, standing; stationary: sitting, standing | `face_up`, `in_pocket`, `in_use`, `on_desk` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 8640 |

Formula: 4608 + 1728 + 8640 = **14976**

### HOSPITAL_APPOINTMENT

- **Unique-sample count:** **2052**
- **Allowed `state_current` values:** `commuting_drive_out`, `commuting_transit_out`, `at_health`
- **Allowed `precondition` values:** `commuting_drive_out`, `commuting_transit_out`, `at_health`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `0`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `1`; `sms_ride_pending` = `0`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `commuting_drive_out` | `commuting_drive_out`, `commuting_transit_out`, `at_health` | `dawn`, `morning` | `workday` | driving: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 54 |
| `commuting_transit_out` | `commuting_drive_out`, `commuting_transit_out`, `at_health` | `dawn`, `morning` | `workday` | transit: sitting | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal` | `cellular` | `0` | 54 |
| `at_health` | `commuting_drive_out`, `commuting_transit_out`, `at_health` | `afternoon`, `dawn`, `evening`, `forenoon`, `late_night`, `lunch`, `morning`, `night`, `sleeping` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `normal` | `normal` | `cellular`, `wifi` | `0` | 1944 |

Formula: 54 + 54 + 1944 = **2052**

### RIDESHARE_PICKUP

- **Unique-sample count:** **4320**
- **Allowed `state_current` values:** `outdoor_walking`, `at_social`, `at_transit_hub`
- **Allowed `precondition` values:** `outdoor_walking`, `at_social`, `at_transit_hub`
- **Fixed relevant categorical constraints (factor = 1):** `cal_hasUpcoming` = `1`; `wifiLost` = `0`; `wifiLostCategory` = `unknown`; `cal_inMeeting` = `0`; `cal_nextLocation` = `unknown`; `sms_delivery_pending` = `0`; `sms_train_pending` = `0`; `sms_flight_pending` = `0`; `sms_hotel_pending` = `0`; `sms_movie_pending` = `0`; `sms_hospital_pending` = `0`; `sms_ride_pending` = `1`

| state_current branch | precondition values | ps_time | ps_dayType | compatible `(ps_motion × activityState)` | ps_phone | ps_light | ps_sound | networkType | isCharging | branch count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `outdoor_walking` | `outdoor_walking`, `at_social`, `at_transit_hub` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | walking: active, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `normal`, `quiet` | `cellular` | `0` | 1296 |
| `at_social` | `outdoor_walking`, `at_social`, `at_transit_hub` | `evening` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `dim`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 432 |
| `at_transit_hub` | `outdoor_walking`, `at_social`, `at_transit_hub` | `afternoon`, `evening`, `forenoon`, `morning` | `holiday`, `weekend`, `workday` | stationary: sitting, standing | `face_up`, `in_pocket`, `in_use` | `bright`, `dim`, `normal` | `noisy`, `normal` | `cellular`, `wifi` | `0` | 2592 |

Formula: 1296 + 432 + 2592 = **4320**


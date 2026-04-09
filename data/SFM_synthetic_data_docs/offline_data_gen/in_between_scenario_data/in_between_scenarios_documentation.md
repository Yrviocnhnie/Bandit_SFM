# In-Between Scenario Evaluation Spec
This document defines 10 evaluation-only in-between scenarios for testing contextual-bandit generalization beyond the original 65 named scenarios.


Design rules used here:
- no new feature classes are introduced
- no new global RO actions are introduced
- each scenario is built only from existing state codes, rule fields, app categories, and profile fields
- important features are listed explicitly, while remaining features must still be adjusted according to `state_current` and realistic context
- `scenarioId` remains metadata only; it is not part of the model input

Interpolation scenarios: between two existing scenarios with overlapping states.
Compositional scenarios: combine two known intents that never appeared together as one named scenario.
Boundary scenarios: same feature classes, but in a timing / transition zone where rigid rules become brittle.

## QUIET_CAFE_BEFORE_MEETING — Quiet cafe before meeting
- **Type:** `compositional`
- **Category:** `work`
- **Parent scenarios:** `CAFE_QUIET`, `MEETING_UPCOMING`, `CAL_HAS_EVENTS`
- **Why this is a good test:** A rigid rule system usually chooses either quiet-cafe behavior or meeting-prep behavior. This new scenario composes both intents in one state and tests whether the model can use calendar context inside a leisure-like place.
- **Allowed `state_current`:** `at_cafe_quiet`
- **Allowed `precondition`:** `office_working`, `office_working_focused`, `home_daytime_workday`
- **Allowed `ps_time`:** `forenoon`, `lunch`, `afternoon`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 1, "cal_inMeeting": 0}`
- **Allowed `cal_eventCount`:** 1, 2, 3
- **Allowed `cal_nextLocation`:** `work`, `education`
- **SMS generation mode:** `none`
- **Default RO ranking:** `O_SHOW_MEETING_DETAILS`, `R_PREPARE_FOR_MEETING`, `O_TURN_ON_SILENT_MODE`
- **Default App ranking:** `productivity`, `navigation`, `reading`
- **Remaining-feature alignment note:** The scenario keeps the state at a quiet cafe, so sound must stay quiet. The calendar is the key extra signal. Remaining features should stay cafe-realistic: stationary or light walking, cafe location, phone on-desk/face-up/in-use patterns, and mostly wifi with some cellular.

## GYM_TO_CAFE_RECOVERY — Gym-to-cafe recovery
- **Type:** `compositional`
- **Category:** `exercise`
- **Parent scenarios:** `HOME_AFTER_GYM`, `OFFICE_TO_CAFE`, `CAFE_STAY`
- **Why this is a good test:** This is a realistic post-workout decompression state, but it is neither a pure gym state nor a pure cafe leisure state. Rules would usually classify one parent and lose the other intent.
- **Allowed `state_current`:** `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition`:** `at_gym`, `at_gym_exercising`
- **Allowed `ps_time`:** `afternoon`, `evening`
- **Allowed `ps_dayType`:** `workday`, `weekend`, `holiday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 0, "cal_eventCount": 0, "cal_inMeeting": 0, "wifiLost": 0, "wifiLostCategory": "unknown"}`
- **Calendar mode:** `none`
- **Allowed `cal_nextLocation`:** `unknown`
- **SMS generation mode:** `none`
- **Default RO ranking:** `R_POST_WORKOUT_RECOVERY`, `R_ENJOY_LEISURE_MOMENT`, `O_SHOW_PAYMENT_QR`
- **Default App ranking:** `health`, `music`, `social`
- **Remaining-feature alignment note:** The recovery intent is represented by the gym precondition, not by inventing a new recovery state. Remaining features must stay cafe-consistent, but battery and activity-duration can reflect that the person has recently been active.

## HOME_DEEP_WORK_MORNING — Deep work morning at home
- **Type:** `interpolation`
- **Category:** `work`
- **Parent scenarios:** `WEEKDAY_HOME_DAY`, `NO_MEETINGS`, `CAL_HAS_EVENTS`
- **Why this is a good test:** There is no exact named scenario for a clean remote deep-work block at home in the morning. Rules can identify home-work or no-meeting day, but they do not express the combined intent as one scenario.
- **Allowed `state_current`:** `home_morning_workday`, `home_daytime_workday`
- **Allowed `precondition`:** `home_sleeping`, `home_sleeping_lying`, `home_morning_workday`
- **Allowed `ps_time`:** `dawn`, `morning`, `forenoon`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 0, "cal_eventCount": 0, "cal_inMeeting": 0, "wifiLost": 0, "wifiLostCategory": "unknown"}`
- **Calendar mode:** `none`
- **Allowed `cal_nextLocation`:** `unknown`
- **SMS generation mode:** `none`
- **Default RO ranking:** `O_SHOW_TODAY_TODO`, `R_DEEP_WORK_WINDOW`, `O_SHOW_SCHEDULE`
- **Default App ranking:** `productivity`, `reading`, `music`
- **Remaining-feature alignment note:** This is a home-work scenario without meeting pressure. Remaining features should stay home-work realistic: mostly stationary, quiet or normal sound, wifi-dominant network, and phone on-desk / in-use / face-up patterns rather than commute or social patterns.

## LATE_OFFICE_PRE_DEPARTURE_WRAPUP — Late office pre-departure wrap-up
- **Type:** `boundary`
- **Category:** `work`
- **Parent scenarios:** `OFFICE_LONG_SESSION`, `LATE_NIGHT_OVERTIME`, `LEAVE_OFFICE`
- **Why this is a good test:** Rules often wait until the departure trigger has already happened or until a long fixed duration has elapsed. This scenario tests whether the bandit can prefer wrap-up behavior earlier from context alone.
- **Allowed `state_current`:** `office_overtime`, `office_late_overtime`
- **Allowed `precondition`:** `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy`
- **Allowed `ps_time`:** `evening`, `night`, `late_night`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 0, "cal_eventCount": 0, "cal_inMeeting": 0, "wifiLost": 0, "wifiLostCategory": "unknown"}`
- **Calendar mode:** `none`
- **Allowed `cal_nextLocation`:** `home`, `unknown`
- **SMS generation mode:** `none`
- **Default RO ranking:** `R_WRAP_UP_AND_REST_AFTER_WORK`, `O_SHOW_TODAY_TODO`, `R_CLOCK_OUT_BEFORE_LEAVING`
- **Default App ranking:** `productivity`, `health`, `music`
- **Remaining-feature alignment note:** Keep the user still inside the office context; the departure intent is represented by time-of-day and office-history preconditions, not by wifi-loss. Remaining features should look like late office work rather than commute.


## TRANSIT_WAIT_WITH_UPCOMING_MEETING — Transit wait with upcoming meeting
- **Type:** `compositional`
- **Category:** `commute`
- **Parent scenarios:** `ARRIVE_TRANSIT_HUB`, `MEETING_UPCOMING`, `TRAIN_DEPARTURE`
- **Why this is a good test:** A rules engine tends to classify this as either transit waiting or meeting preparation. The new scenario composes those intents by keeping the travel-state while making meeting-related context visible.
- **Allowed `state_current`:** `at_metro`, `at_rail_station`, `at_transit_hub`
- **Allowed `precondition`:** `commuting_transit_out`, `commuting_walk_out`, `commuting_drive_out`
- **Allowed `ps_time`:** `morning`, `forenoon`, `afternoon`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 1, "cal_inMeeting": 0}`
- **Allowed `cal_eventCount`:** 1, 2, 3
- **Allowed `cal_nextLocation`:** `work`, `education`
- **SMS generation mode:** `none`
- **Default RO ranking:** `O_SHOW_MEETING_DETAILS`, `R_PREPARE_FOR_MEETING`, `O_SHOW_TRAVEL_INFO`
- **Default App ranking:** `navigation`, `productivity`, `news`
- **Remaining-feature alignment note:** Keep the user stationary at the transit node. Remaining features should stay hub-realistic: cellular network, normal/noisy sound, face-up / in-use / pocket phone states, and sitting or standing activity states.




## RETURN_OFFICE_AFTER_COFFEE — Return to office after coffee
- **Type:** `interpolation`
- **Category:** `work`
- **Parent scenarios:** `ARRIVE_OFFICE`, `CAFE_STAY`, `RETURN_OFFICE_AFTER_LUNCH`
- **Why this is a good test:** No exact legacy rule captures the transition from a short cafe stop back into work mode. Exact rules tend to fire either a cafe leisure action or a generic office-arrival action, but not a context-aware return-to-work action.
- **Allowed `state_current`:** `office_arriving`, `office_working`
- **Allowed `precondition`:** `at_cafe`, `at_cafe_quiet`
- **Allowed `ps_time`:** `morning`, `forenoon`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"wifiLost": 0, "wifiLostCategory": "unknown", "cal_inMeeting": 0}`
- **Calendar mode:** `optional_light`
- **Allowed `cal_nextLocation`:** `unknown`, `work`
- **SMS generation mode:** `none`
- **Default RO ranking:** `O_SHOW_TODAY_TODO`, `O_SHOW_SCHEDULE`, `R_WORK_START_SETTLE_IN`
- **Default App ranking:** `productivity`, `music`, `news`
- **Remaining-feature alignment note:** Important features are return-to-work state, cafe precondition, workday, and morning/forenoon timing. Remaining features should follow office state alignment: work location, realistic office phone postures, wifi/cellular mix, and activity states consistent with stationary/walking office motion.


## POST_LUNCH_WALK_TO_OFFICE — Post-lunch walk back to office
- **Type:** `boundary`
- **Category:** `work`
- **Parent scenarios:** `OFFICE_LUNCH_OUT`, `AFTER_LUNCH_WALK`, `RETURN_OFFICE_AFTER_LUNCH`
- **Why this is a good test:** This is between lunch-out, walking, and returning-to-office. Exact rules often split those cases and miss the mixed intent of an after-meal walk that is already transitioning back toward work.
- **Allowed `state_current`:** `outdoor_walking`
- **Allowed `precondition`:** `at_restaurant_lunch`, `at_cafe`
- **Allowed `ps_time`:** `lunch`, `afternoon`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"wifiLost": 0, "wifiLostCategory": "unknown", "cal_inMeeting": 0}`
- **Calendar mode:** `optional_light`
- **Allowed `cal_nextLocation`:** `work`
- **SMS generation mode:** `none`
- **Default RO ranking:** `R_AFTER_MEAL_WALK`, `O_SHOW_WALKING_ROUTE`, `O_SHOW_SCHEDULE`
- **Default App ranking:** `navigation`, `health`, `music`
- **Remaining-feature alignment note:** Keep the state outdoor_walking, so location must stay outdoor and motion must stay walking. The office-return intent is represented by the lunch-related precondition and work next-location rather than a new state taxonomy.




## UNKNOWN_LONG_STAY_WITH_BOOKING — Unknown long stay with booking signal
- **Type:** `compositional`
- **Category:** `other`
- **Parent scenarios:** `UNKNOWN_LONG_STAY`, `TRAIN_DEPARTURE`, `RIDESHARE_PICKUP`, `HOTEL_CHECKIN`
- **Why this is a good test:** Unknown-long-stay and booking/travel reminder are separate exact rules in the old world. This scenario composes them to see whether the model can prioritize booking-related actions even when location semantics are weak.
- **Allowed `state_current`:** `unknown_settled`, `stationary_unknown`
- **Allowed `precondition`:** `at_transit_hub`, `in_transit`, `commuting_transit_out`
- **Allowed `ps_time`:** `forenoon`, `afternoon`, `evening`, `night`
- **Allowed `ps_dayType`:** `workday`, `weekend`, `holiday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 0, "cal_eventCount": 0, "cal_inMeeting": 0, "wifiLost": 0, "wifiLostCategory": "unknown"}`
- **Calendar mode:** `none`
- **Allowed `cal_nextLocation`:** `unknown`
- **SMS generation mode:** `one_of_booking`
- **Default RO ranking:** `R_CHECK_TRIP_OR_BOOKING_INFO`, `O_SHOW_BOOKING_DETAILS`, `O_PROMPT_ADD_FREQUENT_PLACE`
- **Default App ranking:** `navigation`, `social`, `shopping`
- **Remaining-feature alignment note:** Exactly one booking-like SMS signal should usually be active in a sample. Remaining features should stay unknown-location realistic: stationary motion, quiet/normal/noisy sound, dim/normal light, and a heavier chance of charging or on-desk phone posture.

## HOME_DARK_WITH_CALENDAR_EVENT — Dark home context with calendar event
- **Type:** `compositional`
- **Category:** `home`
- **Parent scenarios:** `HOME_DARK_LONG`, `HOME_EVENING_DARK`, `CAL_HAS_EVENTS`
- **Why this is a good test:** Rules generally treat dark-home context and calendar reminder as different exact situations. This scenario composes both so the right action may involve environment adjustment and schedule surfacing together.
- **Allowed `state_current`:** `home_daytime_workday_dark`, `home_evening_dark`
- **Allowed `precondition`:** `home_daytime_workday`, `home_evening`, `home_daytime_workday_lying`
- **Allowed `ps_time`:** `afternoon`, `evening`, `night`
- **Allowed `ps_dayType`:** `workday`, `weekend`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 1, "cal_inMeeting": 0}`
- **Allowed `cal_eventCount`:** 1, 2, 3
- **Allowed `cal_nextLocation`:** `work`, `education`, `unknown`
- **SMS generation mode:** `none`
- **Default RO ranking:** `R_OPEN_CURTAINS_OR_LIGHT`, `O_SHOW_SCHEDULE`, `O_TURN_ON_EYE_COMFORT_MODE`
- **Default App ranking:** `health`, `reading`, `music`
- **Remaining-feature alignment note:** The chosen dark home substate must force `ps_light=dark`. Remaining features should stay home-realistic rather than commute-like, with a home location, mostly stationary motion, and quiet/normal sound.

## AFTER_WORK_OUTDOOR_RESET — After-work outdoor reset walk
- **Type:** `boundary`
- **Category:** `commute`
- **Parent scenarios:** `LEAVE_OFFICE`, `PARK_WALK`, `LONG_OUTDOOR_WALK`
- **Why this is a good test:** The user has left work, but the immediate intent is not a pure commute and not a pure exercise session. Exact rules often fire one parent scenario and miss the decompression intent.
- **Allowed `state_current`:** `outdoor_walking`
- **Allowed `precondition`:** `office_working`, `office_overtime`, `office_working_focused`, `office_working_noisy`
- **Allowed `ps_time`:** `evening`, `night`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 0, "cal_eventCount": 0, "cal_inMeeting": 0, "wifiLost": 1, "wifiLostCategory": "work"}`
- **Calendar mode:** `none`
- **Allowed `cal_nextLocation`:** `home`, `unknown`
- **SMS generation mode:** `none`
- **Default RO ranking:** `R_ENJOY_LEISURE_MOMENT`, `O_SHOW_WALKING_ROUTE`, `R_OUTDOOR_HYDRATE_AND_REST`
- **Default App ranking:** `navigation`, `health`, `music`
- **Remaining-feature alignment note:** Keep the outdoor-walking state and a work-related precondition. Remaining features should look like a decompression walk rather than intense exercise: outdoor location, walking motion, normal/quiet sound, and mostly cellular connectivity.


# In-Between Scenario Evaluation Spec

This document defines the **final 10 evaluation-only in-between scenarios** for testing contextual-bandit generalization beyond the original 65 named scenarios.

Design rules used here:
- no new feature classes are introduced
- no new `state_current` values are introduced
- no new precondition labels are introduced
- recommendations are chosen only from the **existing shared global RO/action space** and the existing 10 app categories
- important features are listed explicitly, while remaining features must still be adjusted according to `state_current`, substate triggers, and realistic context
- `scenarioId` remains metadata only; it is not part of the model input

---

## QUIET_CAFE_BEFORE_MEETING — Quiet cafe before meeting
- **Type:** `compositional`
- **Category:** `work`
- **Parent scenarios:** `CAFE_QUIET`, `MEETING_UPCOMING`, `CAL_HAS_EVENTS`
- **Why this is a good test:** Rigid rules usually choose either quiet-cafe behavior or meeting-prep behavior. This scenario composes both intents in one state and tests whether the model can prioritize meeting readiness inside a leisure-like place.
- **Allowed `state_current`:** `at_cafe_quiet`
- **Allowed `precondition`:** `office_working`, `office_working_focused`, `home_daytime_workday`
- **Allowed `ps_time`:** `forenoon`, `lunch`, `afternoon`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 1, "cal_inMeeting": 0}`
- **Allowed `cal_eventCount`:** `1`, `2`, `3`
- **Allowed `cal_nextLocation`:** `work`, `education`
- **SMS generation mode:** `none`
- **Default RO ranking:** `meeting_upcoming_details`, `meeting_prep`, `meeting_upcoming_dnd`
- **Default App ranking:** `productivity`, `navigation`, `reading`
- **Remaining-feature alignment note:** Keep the state at a quiet cafe, so sound must stay quiet. Remaining features should stay cafe-realistic: cafe location, stationary or light walking, mostly wifi, and phone states like `on_desk`, `face_up`, or `in_use`.

## GYM_TO_CAFE_RECOVERY — Gym-to-cafe recovery
- **Type:** `compositional`
- **Category:** `exercise`
- **Parent scenarios:** `GYM_WORKOUT`, `GYM_REST`, `CAFE_STAY`
- **Why this is a good test:** This is a realistic post-workout decompression state, but it is neither a pure gym state nor a pure cafe leisure state. Rules would usually classify one parent and lose the other intent.
- **Allowed `state_current`:** `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition`:** `at_gym`, `at_gym_exercising`
- **Allowed `ps_time`:** `afternoon`, `evening`
- **Allowed `ps_dayType`:** `workday`, `weekend`, `holiday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 0, "cal_eventCount": 0, "cal_inMeeting": 0, "wifiLost": 0, "wifiLostCategory": "unknown"}`
- **Calendar mode:** `none`
- **Allowed `cal_nextLocation`:** `unknown`
- **SMS generation mode:** `none`
- **Default RO ranking:** `gym_cooldown`, `cafe_relax`, `cafe_stay_payment`
- **Default App ranking:** `health`, `music`, `social`
- **Remaining-feature alignment note:** The recovery intent is represented by the gym precondition, not by inventing a new state. Remaining features must stay cafe-consistent, but battery and activity duration can reflect that the person has recently been active.

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
- **Default RO ranking:** `weekday_home_day_todo_list`, `weekday_home_day_schedule`, `quiet_home_eye_comfort`
- **Default App ranking:** `productivity`, `reading`, `music`
- **Remaining-feature alignment note:** Keep the scenario home-work realistic: mostly stationary, quiet/normal sound, wifi-dominant network, and phone `on_desk` / `in_use` / `face_up` rather than commute or social patterns.

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
- **Default RO ranking:** `late_night_overtime_rest_and_wrap_up`, `office_afternoon_todo_list`, `leave_office_clock_out`
- **Default App ranking:** `productivity`, `health`, `music`
- **Remaining-feature alignment note:** Keep the user still inside the office context; the departure intent is represented by time-of-day and office-history preconditions, not by `wifiLost`.

## HOME_WORK_DELIVERY_INTERRUPT — Work-from-home delivery interrupt
- **Type:** `compositional`
- **Category:** `home`
- **Parent scenarios:** `WEEKDAY_HOME_DAY`, `DELIVERY_AT_OFFICE`, `NO_MEETINGS`
- **Why this is a good test:** Rules can identify home-working context and package-delivery context independently, but they do not express the interruption of a focused work block by a delivery event. The combined context changes the right ranking.
- **Allowed `state_current`:** `home_daytime_workday`, `home_daytime_workday_dark`, `home_morning_workday`
- **Allowed `precondition`:** `home_morning_workday`, `home_daytime_workday`, `home_sleeping`
- **Allowed `ps_time`:** `morning`, `forenoon`, `afternoon`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"cal_inMeeting": 0, "wifiLost": 0, "wifiLostCategory": "unknown", "sms_delivery_pending": 1}`
- **Calendar mode:** `optional_light`
- **Allowed `cal_nextLocation`:** `home`, `unknown`
- **SMS generation mode:** `delivery_only`
- **Default RO ranking:** `delivery_pickup`, `delivery_at_office_show_pickup_info`, `weekday_home_day_todo_list`
- **Default App ranking:** `shopping`, `productivity`, `social`
- **Remaining-feature alignment note:** Keep the state home-work aligned. Remaining features should look like a daytime home-work block: home location, mostly stationary, wifi-dominant network, and phone on desk / in use. The delivery interrupt is carried by the SMS field rather than a new taxonomy value.

## OFFICE_TO_GYM_TRANSITION — Leaving office to go to the gym
- **Type:** `compositional`
- **Category:** `transition`
- **Parent scenarios:** `LEAVE_OFFICE`, `HOME_TO_GYM`, `GYM_WORKOUT`
- **Why this is a good test:** This is a very practical product case. The user is leaving work, but the next intent is not home commute; it is exercise. A rigid rule often overcommits to commute-home behavior.
- **Allowed `state_current`:** `commuting_walk_out`, `commuting_drive_out`, `outdoor_walking`
- **Allowed `precondition`:** `office_working`, `office_working_focused`, `office_overtime`
- **Allowed `ps_time`:** `evening`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"wifiLost": 1, "wifiLostCategory": "work", "cal_hasUpcoming": 0, "cal_eventCount": 0, "cal_inMeeting": 0}`
- **Allowed `cal_nextLocation`:** `gym`
- **SMS generation mode:** `none`
- **Default RO ranking:** `exercise_tracking`, `home_to_gym_timer`, `leave_office_weather`
- **Default App ranking:** `health`, `music`, `navigation`
- **Remaining-feature alignment note:** Keep the transition outdoor/commute realistic. If `state_current` is commute-like, use cellular connectivity and motion-aligned transport. The gym intent should be represented by `cal_nextLocation=gym` and the work-exit signal.

## OFFICE_LUNCH_OUT_WITH_UPCOMING_MEETING — Lunch out with upcoming meeting
- **Type:** `compositional`
- **Category:** `work`
- **Parent scenarios:** `OFFICE_LUNCH_OUT`, `MEETING_UPCOMING`, `CAL_HAS_EVENTS`
- **Why this is a good test:** This is much easier to justify than a transit waiting variant. The user is out for lunch, but the next 20–30 minutes are constrained by an upcoming meeting, so the correct ranking is not just lunch and not just meeting.
- **Allowed `state_current`:** `at_cafe`, `at_cafe_quiet`, `at_restaurant_lunch`
- **Allowed `precondition`:** `office_arriving`, `office_working`, `office_lunch_break`
- **Allowed `ps_time`:** `lunch`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 1, "cal_inMeeting": 0}`
- **Allowed `cal_eventCount`:** `1`, `2`, `3`
- **Allowed `cal_nextLocation`:** `work`
- **SMS generation mode:** `none`
- **Default RO ranking:** `meeting_upcoming_details`, `office_lunch_out_payment`, `meeting_prep`
- **Default App ranking:** `productivity`, `shopping`, `navigation`
- **Remaining-feature alignment note:** Lunch-state features must stay dining/cafe realistic. The meeting urgency is carried by calendar fields and work next-location, not by inventing a new lunch-meeting state.

## WEEKEND_CAFE_OUTING_PLANNING — Weekend cafe before outing planning
- **Type:** `interpolation`
- **Category:** `weekend`
- **Parent scenarios:** `WEEKEND_MORNING`, `WEEKEND_OUTING`, `CAFE_STAY`
- **Why this is a good test:** Weekend-morning rules often default to weather/greeting, while cafe rules default to leisure or reading. This scenario composes a relaxed weekend cafe state with immediate outing-planning intent.
- **Allowed `state_current`:** `at_cafe`, `at_cafe_quiet`
- **Allowed `precondition`:** `home_morning_rest`, `home_morning_rest_lying`, `home_daytime_rest`
- **Allowed `ps_time`:** `morning`, `forenoon`
- **Allowed `ps_dayType`:** `weekend`, `holiday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 0, "cal_eventCount": 0, "cal_inMeeting": 0}`
- **Allowed `cal_nextLocation`:** `outdoor`, `shopping`, `social`, `unknown`
- **SMS generation mode:** `none`
- **Default RO ranking:** `weekend_weather`, `weekend_outing_nearby_places`, `cafe_stay_schedule`
- **Default App ranking:** `navigation`, `social`, `reading`
- **Remaining-feature alignment note:** Keep the state cafe-consistent and relaxed. The outing-planning intent is represented by weekend timing, home-rest preconditions, and a forward-looking next-location rather than a new state code.

## ARRIVE_OFFICE_WITH_IMMINENT_MEETING — Arrive office with imminent meeting
- **Type:** `boundary`
- **Category:** `work`
- **Parent scenarios:** `ARRIVE_OFFICE`, `MEETING_UPCOMING`, `CAL_HAS_EVENTS`
- **Why this is a good test:** Generic office-arrival behavior is not always right. If a meeting is very soon, the best action is meeting prep first. This is easy to explain in a demo and clearly different from standard office-arrival defaults.
- **Allowed `state_current`:** `office_arriving`, `office_working`
- **Allowed `precondition`:** `commuting_walk_out`, `commuting_drive_out`, `commuting_transit_out`
- **Allowed `ps_time`:** `morning`, `forenoon`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"cal_hasUpcoming": 1, "cal_inMeeting": 0}`
- **Allowed `cal_eventCount`:** `1`, `2`
- **Allowed `cal_nextLocation`:** `work`
- **SMS generation mode:** `none`
- **Default RO ranking:** `meeting_upcoming_details`, `meeting_prep`, `arrive_office_schedule`
- **Default App ranking:** `productivity`, `reading`, `news`
- **Remaining-feature alignment note:** Keep the arrival state office-consistent with commute preconditions and work location. The meeting urgency should alter ranking, not the base state encoding.

## LEAVE_OFFICE_WITH_ERRAND_STOP — Leave office with an errand stop
- **Type:** `compositional`
- **Category:** `transition`
- **Parent scenarios:** `LEAVE_OFFICE`, `SHOPPING`, `HEALTHCARE_VISIT` / `CUSTOM_PLACE_VISIT`
- **Why this is a good test:** This is stronger than plain `LEAVE_OFFICE`. The next intent is not home commute, but a short stop for an errand. The work-exit signal should combine with the next-destination context to change the action ranking.
- **Allowed `state_current`:** `commuting_walk_out`, `commuting_drive_out`, `outdoor_walking`
- **Allowed `precondition`:** `office_working`, `office_overtime`, `office_working_focused`
- **Allowed `ps_time`:** `evening`
- **Allowed `ps_dayType`:** `workday`
- **Rule-dependent fixed fields:** `{"wifiLost": 1, "wifiLostCategory": "work", "cal_hasUpcoming": 0, "cal_eventCount": 0, "cal_inMeeting": 0}`
- **Allowed `cal_nextLocation`:** `shopping`, `health`, `custom`
- **SMS generation mode:** `none`
- **Default RO ranking:** `leave_office_weather`, `shopping_payment`, `park_walk_route`
- **Default App ranking:** `navigation`, `shopping`, `health`
- **Remaining-feature alignment note:** Keep the user in a realistic post-office transition state. The errand intent is represented by the next-location and work-exit signal, while the remaining features stay consistent with commute/outdoor movement.

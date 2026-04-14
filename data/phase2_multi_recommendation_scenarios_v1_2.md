# Phase-2 Multi-Anchor Feedback Scenarios v1.2

This document extends `phase2_multi_recommendation_scenarios_v1.md` for
phase-2 online learning simulation.

Goal:
- keep the original per-context `like` feedback
- add one simulated `dislike` feedback per context
- make the dislike action come from the current offline model's displayed
  `top-3` actions whenever possible

Construction rule:
1. Use the current best offline R/O model
   (`v6_2k_neural_linear_3p3a_hardneg_other_stratified_split`) to infer the
   `top-3` actions for each context.
2. Keep the original `Recommendation` as the simulated `like`.
3. Excluding the liked action, choose one of the other shown `top-3` actions as
   the simulated `dislike`.
4. When the liked action is not in the inferred baseline `top-3`, keep the
   original `like` for phase-2 experimentation but record the caveat explicitly.

Notes:
- The inferred top-3 below are based on the offline model plus the feature
  fields explicitly listed in each context, with stable defaults used for
  unspecified fields.
- This file is intended for phase-2 online-learning simulation, where each
  feedback item becomes one anchor and propagates to its local latent neighbors.

---

## 1. SCENARIO: ARRIVE_OFFICE

### Context A: Meeting-heavy arrival

**Features**
- state_current: office_arriving
- precondition: commuting_transit_out
- ps_time: morning
- ps_dayType: workday
- ps_motion: walking
- ps_location: work
- cal_hasUpcoming: 1
- cal_eventCount: 3
- cal_inMeeting: 0

**Offline baseline top-3**
1. `O_SHOW_SCHEDULE`
2. `O_SHOW_TODAY_TODO`
3. `R_PLAN_DAY_OVER_COFFEE`

**Simulated feedback**
- like: `O_SHOW_SCHEDULE`
- dislike: `R_PLAN_DAY_OVER_COFFEE`

**Reason**
- like: high meeting density makes schedule awareness most useful.
- dislike: among the other shown actions, `R_PLAN_DAY_OVER_COFFEE` is the least
  urgent in this meeting-heavy arrival context.

---

### Context B: Task-focused start

**Features**
- state_current: office_working
- precondition: commuting_walk_out
- ps_time: forenoon
- ps_dayType: workday
- ps_motion: stationary
- ps_location: work
- ps_phone: on_desk
- activityState: sitting
- cal_hasUpcoming: 0
- cal_eventCount: 0

**Offline baseline top-3**
1. `O_SHOW_TODAY_TODO`
2. `O_SHOW_SCHEDULE`
3. `R_DEEP_WORK_WINDOW`

**Simulated feedback**
- like: `O_SHOW_TODAY_TODO`
- dislike: `R_DEEP_WORK_WINDOW`

**Reason**
- like: no meetings means the task list is the most actionable.
- dislike: among the displayed alternatives, a generic deep-work reminder is
  less directly useful than surfacing concrete tasks or schedule.

---

### Context C: Soft start morning

**Features**
- state_current: office_arriving
- precondition: commuting_walk_out
- ps_time: dawn
- ps_dayType: workday
- ps_motion: walking
- ps_phone: in_pocket
- cal_hasUpcoming: 0

**Offline baseline top-3**
1. `O_SHOW_TODAY_TODO`
2. `O_SHOW_SCHEDULE`
3. `O_ENABLE_VIBRATION`

**Simulated feedback**
- like: `R_PLAN_DAY_OVER_COFFEE`
- dislike: `O_ENABLE_VIBRATION`

**Reason**
- like: low-urgency soft-start preference is the intended personalized signal.
- dislike: `O_ENABLE_VIBRATION` is the least reasonable of the displayed
  alternatives for a calm early-arrival context.

**Caveat**
- The liked action `R_PLAN_DAY_OVER_COFFEE` is not in the current inferred
  offline baseline top-3 for this partially specified context. We keep it here
  intentionally as a phase-2 preference-override case.

---

## 2. SCENARIO: LEAVE_OFFICE

### Context A: Driving commute

**Features**
- state_current: commuting_drive_home
- precondition: office_overtime
- ps_time: night
- ps_dayType: workday
- ps_motion: driving
- ps_location: en_route
- wifiLost: 1
- wifiLostCategory: work
- networkType: cellular

**Offline baseline top-3**
1. `O_SHOW_WEATHER`
2. `O_SHOW_COMMUTE_TRAFFIC`
3. `R_CLOCK_OUT_BEFORE_LEAVING`

**Simulated feedback**
- like: `O_SHOW_COMMUTE_TRAFFIC`
- dislike: `R_CLOCK_OUT_BEFORE_LEAVING`

**Reason**
- like: traffic is the most actionable signal during a real drive home.
- dislike: once already en route, a clock-out reminder is less useful than the
  other displayed options.

---

### Context B: Walking commute

**Features**
- state_current: outdoor_walking
- precondition: office_working
- ps_time: evening
- ps_dayType: workday
- ps_motion: walking
- ps_location: outdoor
- wifiLost: 1
- wifiLostCategory: work

**Offline baseline top-3**
1. `O_SHOW_WEATHER`
2. `R_CLOCK_OUT_BEFORE_LEAVING`
3. `O_SHOW_COMMUTE_TRAFFIC`

**Simulated feedback**
- like: `O_SHOW_WEATHER`
- dislike: `O_SHOW_COMMUTE_TRAFFIC`

**Reason**
- like: for a walking commute, weather is the most directly useful signal.
- dislike: among the other shown actions, traffic is the least relevant for a
  non-driving return home.

---

## 3. SCENARIO: OFFICE_LUNCH_OUT

### Context A: Ready to pay

**Features**
- state_current: at_restaurant_lunch
- precondition: office_working
- ps_time: lunch
- ps_dayType: workday
- ps_motion: stationary
- ps_location: restaurant
- wifiLost: 1
- wifiLostCategory: work
- activityState: sitting

**Offline baseline top-3**
1. `O_SHOW_NEARBY_OPTIONS`
2. `R_MEAL_BREAK`
3. `O_SHOW_PAYMENT_QR`

**Simulated feedback**
- like: `O_SHOW_PAYMENT_QR`
- dislike: `R_MEAL_BREAK`

**Reason**
- like: the user is at the payment stage, so QR is the best personalized action.
- dislike: compared with nearby options or payment, a generic meal-break
  reminder is the least useful at this exact moment.

---

### Context B: Stepping out / break phase

**Features**
- state_current: outdoor_walking
- precondition: office_lunch_break
- ps_time: lunch
- ps_dayType: workday
- ps_motion: walking
- ps_location: outdoor
- wifiLost: 1
- wifiLostCategory: work
- activityState: active

**Offline baseline top-3**
1. `O_SHOW_NEARBY_OPTIONS`
2. `O_SHOW_PAYMENT_QR`
3. `R_MEAL_BREAK`

**Simulated feedback**
- like: `R_ENJOY_LEISURE_MOMENT`
- dislike: `R_MEAL_BREAK`

**Reason**
- like: the user is explicitly in a break-oriented transition phase.
- dislike: among the shown top-3, `R_MEAL_BREAK` is less aligned than either
  nearby options or payment utility for this particular stepping-out moment.

**Caveat**
- The liked action `R_ENJOY_LEISURE_MOMENT` is not in the current inferred
  offline baseline top-3 for this partially specified context. We keep it here
  intentionally as a phase-2 preference-override case.

---

## 4. SCENARIO: CAFE_QUIET

### Context A: Upcoming meeting

**Features**
- state_current: at_cafe_quiet
- precondition: at_cafe
- ps_time: forenoon
- ps_dayType: workday
- ps_motion: stationary
- ps_location: cafe
- ps_sound: quiet
- cal_hasUpcoming: 1
- cal_eventCount: 1
- cal_nextLocation: work

**Offline baseline top-3**
1. `O_SHOW_TRAVEL_INFO`
2. `R_DEPART_EARLY_FOR_OFFSITE`
3. `O_SHOW_MEETING_DETAILS`

**Simulated feedback**
- like: `O_SHOW_MEETING_DETAILS`
- dislike: `R_DEPART_EARLY_FOR_OFFSITE`

**Reason**
- like: meeting details are the most precise action for this quiet pre-meeting
  cafe context.
- dislike: among the other displayed actions, an early-departure reminder is
  less directly appropriate than concrete meeting info.

---

### Context B: Deep focus / low interruption

**Features**
- state_current: at_cafe_quiet
- precondition: at_cafe
- ps_time: afternoon
- ps_dayType: workday
- ps_motion: stationary
- ps_location: cafe
- ps_sound: quiet
- ps_phone: face_up
- cal_hasUpcoming: 0
- cal_eventCount: 0

**Offline baseline top-3**
1. `O_SHOW_SCHEDULE`
2. `R_DEEP_WORK_WINDOW`
3. `O_TURN_ON_SILENT_MODE`

**Simulated feedback**
- like: `O_TURN_ON_SILENT_MODE`
- dislike: `R_DEEP_WORK_WINDOW`

**Reason**
- like: the user prefers protecting a quiet focus environment.
- dislike: among the other displayed actions, the generic deep-work reminder is
  less concrete than explicitly enabling silent mode.

---

## Summary

This v1.2 file is designed for phase-2 online-learning simulation:

- Each scenario now contains multiple feedback anchors.
- Each context has an explicit `like` and `dislike` feedback item.
- The dislike item is chosen from the current offline model's displayed `top-3`
  whenever possible.
- Two contexts are intentionally kept as preference-override cases, where the
  liked action is not in the current baseline top-3:
  - `ARRIVE_OFFICE / Context C`
  - `OFFICE_LUNCH_OUT / Context B`

These cases are useful for showing that phase-2 online learning can learn
different user preferences for different contexts within the same scenario.

# Contextual Scenario Variations with Feature-Based Recommendations

This document defines a set of real-world scenarios with varying contexts.
Each context specifies key input feature values and the corresponding
recommended RO (Reminder/Operation) action.


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

**Recommendation**
- O_SHOW_SCHEDULE

**Reason**
High meeting density → schedule awareness is critical.

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

**Recommendation**
- O_SHOW_TODAY_TODO

**Reason**
No meetings → task execution becomes priority.

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

**Recommendation**
- R_PLAN_DAY_OVER_COFFEE

**Reason**
Low urgency → routine-based recommendation fits better.

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

**Recommendation**
- O_SHOW_COMMUTE_TRAFFIC

**Reason**
Driving → traffic/navigation is most useful.

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

**Recommendation**
- O_SHOW_WEATHER

**Reason**
Still a commute context → navigation remains relevant.

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

**Recommendation**
- O_SHOW_PAYMENT_QR

**Reason**
User is at payment stage → QR/payment is optimal.

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

**Recommendation**
- R_ENJOY_LEISURE_MOMENT

**Reason**
Transition phase → break-oriented recommendation fits.

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

**Recommendation**
- O_SHOW_MEETING_DETAILS

**Reason**
Meeting context dominates → show meeting details.

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

**Recommendation**
- O_TURN_ON_SILENT_MODE

**Reason**
Quiet + no schedule pressure → protect focus environment.

---

## Summary

This document demonstrates:

- Same scenario → different contexts → different recommendations
- Context is driven by feature variations:
  - calendar signals
  - motion
  - time
  - preconditions
  - environment (sound/light)
- Shows how a contextual bandit can outperform rigid rules by adapting to feature combinations.

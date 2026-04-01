# Added `state_current` for scenarios that previously had none

Total scenarios updated: **13**

| scenarioId | scenarioNameEn | added state_current values | rationale |
| --- | --- | --- | --- |
| `OFFICE_LUNCH_OUT` | Office lunch out | `outdoor_walking`<br>`at_restaurant_lunch`<br>`at_cafe`<br>`at_cafe_quiet` | Use lunch-out states after leaving office Wi-Fi: walking out, restaurant lunch, or café. |
| `LEAVE_OFFICE` | Leave office | `commuting_walk_home`<br>`commuting_cycle_home`<br>`commuting_drive_home`<br>`commuting_transit_home`<br>`outdoor_walking` | Use homebound commute states after leaving office in the evening. |
| `DELIVERY_AT_OFFICE` | Package arrived at office | `office_arriving`<br>`office_working`<br>`office_working_focused`<br>`office_working_noisy` | Keep current state in office contexts because the package reminder is received while at work. |
| `MEETING_UPCOMING` | Meeting starting soon | `office_working`<br>`office_working_focused`<br>`home_daytime_workday`<br>`at_cafe_quiet` | Keep current state in the working context where the user is preparing for the upcoming meeting. |
| `IN_MEETING` | In meeting | `office_working_focused`<br>`home_daytime_workday`<br>`at_cafe_quiet` | Use focused/stationary working contexts that realistically correspond to being in a meeting. |
| `REMOTE_MEETING` | Remote meeting departure | `home_daytime_workday`<br>`at_cafe_quiet` | Keep current state at the origin context before departure; the travel/offsite location is captured by calendar fields. |
| `CAL_HAS_EVENTS` | Today's schedule reminder | `office_working`<br>`office_working_focused`<br>`home_daytime_workday`<br>`at_cafe_quiet` | Use broad working-day contexts where a schedule reminder makes sense. |
| `TRAIN_DEPARTURE` | Train departure reminder | `commuting_transit_out`<br>`at_transit_hub`<br>`at_rail_station` | Use transit and rail-station states near departure. |
| `FLIGHT_BOARDING` | Flight boarding reminder | `commuting_drive_out`<br>`commuting_transit_out`<br>`at_airport` | Use outbound travel states and airport state near flight boarding. |
| `HOTEL_CHECKIN` | Hotel check-in reminder | `commuting_drive_out`<br>`commuting_transit_out`<br>`at_airport`<br>`at_transit_hub` | Use travel-in-progress states that realistically precede hotel arrival/check-in. |
| `MOVIE_TICKET` | Movie ticket reminder | `home_evening`<br>`at_social`<br>`at_restaurant_other` | Use evening/social contexts that fit movie booking behavior. |
| `HOSPITAL_APPOINTMENT` | Hospital appointment reminder | `commuting_drive_out`<br>`commuting_transit_out`<br>`at_health` | Use outbound travel or health-location states that fit an appointment reminder. |
| `RIDESHARE_PICKUP` | Rideshare pickup reminder | `outdoor_walking`<br>`at_social`<br>`at_transit_hub` | Use outdoor/social/transit-hub states where a rideshare pickup reminder is plausible. |

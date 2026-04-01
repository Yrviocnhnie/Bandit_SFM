# Added Preconditions Summary

Added new `preconditions` to **55** scenarios. The existing **10** scenarios that already had preconditions were left unchanged.

Approach used:
- preconditions use only `state_current` values
- state codes were chosen from realistic immediately preceding contexts
- existing rules were not altered except for adding missing preconditions

| scenarioId | added precondition state_current values | withinMs | rationale |
| --- | --- | ---: | --- |
| `ARRIVE_OFFICE` | `commuting_walk_out,commuting_cycle_out,commuting_drive_out,commuting_transit_out` | 7200000 | Arrive office is typically preceded by outbound commute. |
| `OFFICE_AFTERNOON` | `office_arriving,office_working,office_lunch_break` | 7200000 | Afternoon office state usually follows earlier office presence or lunch break. |
| `WEEKDAY_HOME_DAY` | `home_morning_workday,home_morning_workday_lying` | 10800000 | Weekday daytime at home usually follows the home morning workday state. |
| `ARRIVE_TRANSIT_HUB` | `commuting_walk_out,commuting_cycle_out,commuting_drive_out,commuting_transit_out` | 7200000 | Transit hub arrival is typically preceded by outbound commute. |
| `LATE_NIGHT_OVERTIME` | `office_working,office_working_focused,office_working_noisy,office_overtime` | 10800000 | Late-night overtime follows extended office work. |
| `WEEKEND_OVERTIME` | `office_arriving,office_rest_day` | 7200000 | Weekend overtime usually follows arrival to office on a rest day. |
| `EVENING_AT_OFFICE` | `office_working,office_working_focused,office_working_noisy,office_overtime` | 7200000 | Evening office presence follows earlier office work. |
| `WAKE_UP` | `home_sleeping,home_sleeping_lying` | 1800000 | Wake-up should immediately follow sleeping state. |
| `MORNING_EXERCISE` | `home_morning_workday,home_morning_rest,home_morning_rest_lying` | 3600000 | Morning exercise usually starts from home in the morning. |
| `SHOPPING` | `commuting_drive_out,commuting_walk_out,outdoor_walking` | 7200000 | Shopping is usually preceded by outbound movement. |
| `HOME_EVENING` | `commuting_drive_home,commuting_transit_home,commuting_walk_home,commuting_cycle_home` | 3600000 | Evening at home usually follows inbound commute. |
| `ARRIVE_DINING` | `outdoor_walking,at_shopping,commuting_walk_out,commuting_drive_out` | 3600000 | Dining arrival is usually preceded by walking or travel from another activity. |
| `HOME_LATE_NIGHT` | `home_evening,home_evening_dark,home_evening_lying,home_evening_noisy` | 7200000 | Late-night home state typically follows earlier evening-at-home state. |
| `WEEKEND_MORNING` | `home_sleeping,home_sleeping_lying` | 3600000 | Weekend morning usually follows sleeping. |
| `WEEKEND_OUTING` | `home_morning_rest,home_morning_rest_lying,home_daytime_rest` | 7200000 | Weekend outing usually starts from relaxed home states. |
| `PARK_WALK` | `home_daytime_rest,home_daytime_workday,outdoor_walking` | 3600000 | Park walk is usually preceded by home/rest and then outdoor movement. |
| `LONG_DRIVE` | `commuting_drive_out,driving` | 7200000 | Long drive follows being in a driving/commute state. |
| `HOME_AFTERNOON` | `home_morning_rest,home_morning_workday,home_daytime_workday` | 10800000 | Home afternoon typically follows earlier home-day states. |
| `MORNING_COFFEE` | `home_morning_workday,home_morning_rest,commuting_walk_out` | 3600000 | Morning coffee usually follows leaving home in the morning. |
| `CAFE_STAY` | `at_cafe,at_cafe_quiet` | 3600000 | Cafe stay follows already being at a cafe. |
| `QUIET_HOME` | `home_daytime_workday,home_daytime_rest,home_morning_workday,home_morning_rest,home_evening` | 7200000 | Quiet-home scenario should be preceded by another home state. |
| `GYM_WORKOUT` | `at_gym` | 1800000 | Workout follows arrival or waiting at the gym. |
| `GYM_REST` | `at_gym_exercising` | 1800000 | Gym rest follows a workout state. |
| `GYM_LONG_STAY` | `at_gym,at_gym_exercising` | 7200000 | Long gym stay follows earlier gym presence. |
| `OUTDOOR_RUNNING` | `home_morning_rest,home_morning_workday,outdoor_walking` | 3600000 | Outdoor run usually begins from home or a walk warmup. |
| `NOISY_GATHERING` | `home_evening,at_restaurant_dinner,at_social` | 7200000 | Noisy gathering is usually preceded by evening socializing or dining. |
| `WEEKEND_OUTDOOR_WALK` | `home_morning_rest,home_daytime_rest` | 7200000 | Weekend outdoor walk typically starts from home rest states. |
| `LONG_OUTDOOR_WALK` | `outdoor_walking` | 7200000 | Long walk follows earlier walking. |
| `CYCLING` | `commuting_cycle_out,commuting_cycle_home,outdoor_cycling` | 3600000 | Cycling follows being on a bike or cycling commute. |
| `LATE_NIGHT_NOISY` | `home_evening,at_social,at_restaurant_dinner` | 7200000 | Late-night noisy state follows evening social contexts. |
| `EDUCATION_WAITING` | `commuting_walk_out,commuting_transit_out,at_education` | 3600000 | Education waiting follows travel to an education venue or being there already. |
| `EDUCATION_LONG_SIT` | `at_education,at_education_class` | 7200000 | Long sit in education follows being in an education context. |
| `HOME_LUNCH` | `home_daytime_workday,home_daytime_rest` | 3600000 | Home lunch follows being at home earlier in the day. |
| `AFTER_LUNCH_WALK` | `home_daytime_rest,home_daytime_rest_dark,home_daytime_rest_lying,home_daytime_workday` | 1800000 | After-lunch walk follows a lunch/home-rest state. |
| `DELIVERY_AT_OFFICE` | `office_working,office_arriving` | 7200000 | Delivery at office should be preceded by office presence. |
| `MEETING_UPCOMING` | `office_working,office_working_focused,home_daytime_workday,at_cafe_quiet` | 7200000 | Upcoming meeting usually arises while working in office/home/cafe. |
| `IN_MEETING` | `office_working,office_working_focused,home_daytime_workday,at_cafe_quiet` | 1800000 | In-meeting state should be preceded by being in a meeting-capable work context. |
| `REMOTE_MEETING` | `home_daytime_workday,at_cafe_quiet` | 3600000 | Remote meeting is typically preceded by working at home or in a quiet cafe. |
| `NO_MEETINGS` | `office_arriving,office_working,home_daytime_workday` | 7200000 | No-meetings workday context should follow normal work presence. |
| `CAL_HAS_EVENTS` | `office_working,home_daytime_workday,at_cafe_quiet` | 7200000 | Calendar events are relevant in ordinary work contexts. |
| `LATE_NIGHT_PHONE` | `home_evening,home_evening_lying,home_sleeping` | 7200000 | Late-night phone use follows evening-at-home or pre-sleep states. |
| `OFFICE_FOCUS_LONG` | `office_working,office_working_focused` | 7200000 | Long focus follows sustained office work. |
| `HOME_DARK_LONG` | `home_daytime_workday_dark,home_daytime_rest_dark,home_evening_dark` | 10800000 | Long dark-home state follows earlier dark-at-home states. |
| `UNKNOWN_LONG_STAY` | `stationary_unknown,unknown_settled` | 7200000 | Long unknown stay should follow another unknown settled state. |
| `OFFICE_LONG_SESSION` | `office_working,office_working_focused,office_working_noisy` | 14400000 | Very long office session follows normal office work states. |
| `HOME_EVENING_DARK` | `commuting_drive_home,commuting_transit_home,commuting_walk_home,home_evening` | 3600000 | Home evening dark usually follows commute-home or earlier home-evening. |
| `HOME_EVENING_NOISY` | `home_evening,at_social,at_restaurant_dinner` | 3600000 | Home evening noisy usually follows evening social activity or earlier home evening. |
| `SCHOOL_QUIET` | `at_education,at_education_class` | 3600000 | Quiet school context follows being at school/class. |
| `CAFE_QUIET` | `at_cafe` | 3600000 | Quiet cafe state follows being at a cafe. |
| `TRAIN_DEPARTURE` | `commuting_transit_out,at_transit_hub,at_rail_station` | 7200000 | Train departure follows travel toward a station/hub. |
| `FLIGHT_BOARDING` | `commuting_drive_out,commuting_transit_out,at_airport` | 10800000 | Flight boarding follows travel toward the airport. |
| `HOTEL_CHECKIN` | `commuting_drive_out,commuting_transit_out,at_airport,at_transit_hub` | 21600000 | Hotel check-in follows longer travel movement. |
| `MOVIE_TICKET` | `home_evening,at_social,at_restaurant_other` | 7200000 | Movie-ticket scenario usually follows an evening leisure context. |
| `HOSPITAL_APPOINTMENT` | `commuting_drive_out,commuting_transit_out,at_health` | 7200000 | Hospital appointment follows travel toward a health location. |
| `RIDESHARE_PICKUP` | `outdoor_walking,at_social,at_transit_hub` | 3600000 | Rideshare pickup follows being outdoors or near a pickup location. |

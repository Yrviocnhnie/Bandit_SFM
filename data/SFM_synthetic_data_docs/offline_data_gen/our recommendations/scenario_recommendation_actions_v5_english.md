# v5 Global Shared Action Space

## English Notes

In this v5 version, each scenario no longer maintains its own separate set of R/O actions. Instead, the original **167 legacy R/O actions** are consolidated into a shared **global action space**.

Key changes:
- Total legacy `R/O` actions: **167**
- Total global `R/O` actions: **46**
- Global `R` actions: **26**
- Global `O` actions: **20**
- Total scenarios: **65**
- Each scenario now includes **>=2 default R/O actions**, ordered by common-sense priority.
- Apps still use the fixed **10 app categories**, but each scenario now includes **>=3 default app categories**, ranked by suitability.

Focus of this refactor:
- **How O actions are merged**: merge primarily by operation. If parameters are needed, they must be resolved directly from existing features; the model itself only recommends the action.
- **How R actions are merged**: merge primarily by real user intent, such as "getting into work mode," "taking a break during work," "post-workout recovery," or "checking trip/booking information."

Output files:
- `outputs/global_action_space_details.json`: the full action space file, containing the global R/O action space together with per-scenario mappings, default action order, and default app order.
- `outputs/global_action_space.json`: a lighter action space file that excludes old-to-new mappings and keeps only the global action space plus the default actions/default apps for each scenario.
- The original legacy data is preserved in `references/scenario_actions.json`.

## Merge Principles

- O actions are merged primarily by operation. If parameters are needed, they must be directly resolvable from the current features, and RL/UCB only recommends the action.
- R actions are merged primarily by user intent. If only the wording or scenario phrasing differs, they are consolidated into the same reminder semantics.
- The scenario layer keeps only action references, trigger conditions, and default ordering, and does not carry action parameters; parameters are decided by the feature resolver at execution time.
- Each scenario keeps at least 2 default R/O actions and at least 3 default app category items ranked by common sense.

## Global Reminder Space

| global R actionId | message | merged legacy count | merge note |
| --- | --- | --- | --- |
| `R_AFTER_MEAL_WALK` | A short walk after the meal would feel nice. | 1 | Keeps the standalone reminder for a walk after a meal. |
| `R_CALENDAR_QUICK_LOOK` | It looks like you have plans today. A quick look will help you stay on top of them. | 1 | Keeps the light reminder to take a quick look when the day has schedule items. |
| `R_CAPTURE_MEETING_FOLLOW_UPS` | Stay focused on the discussion, and jot down follow-up items as they come up. | 1 | Keeps the reminder to record follow-up items during meetings. |
| `R_CHECK_TRIP_OR_BOOKING_INFO` | It is a good idea to confirm the key details of this trip or booking first, so things do not feel rushed later. | 7 | Merges reminders to confirm key details in advance for transit departures, hotels, movies, hospital visits, rideshares, and similar cases. |
| `R_CLOCK_OUT_BEFORE_LEAVING` | Before leaving, do not forget to finish the departure-related steps, such as clocking out. | 1 | Keeps the explicit reminder to complete departure actions before leaving the office. |
| `R_COMMUTE_STEADY_AND_SAFE` | Take it steady on the road. No need to rush. Safety comes first. | 2 | Merges "steady and safe" reminders across commuting and cycling. |
| `R_DEEP_WORK_WINDOW` | This is a good stretch of time for focused work on something important. | 3 | Merges "good for focus" reminders across no-meeting days, quiet school periods, and quiet cafés. |
| `R_DELIVERY_PICKUP_REMINDER` | There is something waiting to be handled. Pick it up when it is convenient. | 1 | Keeps the explicit reminder for package or item pickup. |
| `R_DEPART_EARLY_FOR_OFFSITE` | This involves travel, so leaving a bit early will make things more comfortable. | 1 | Keeps the reminder to leave a bit early when travel is needed for a meeting or appointment. |
| `R_ENJOY_LEISURE_MOMENT` | This is a good time to relax a little and simply enjoy the moment. | 7 | Merges light "enjoy the moment" reminders across shopping, dining, coffee, walking, and weekend outings. |
| `R_HOME_RELAX_AND_RESET` | Let yourself slow down, drink some water, relax a bit, and then decide what to do next. | 6 | Merges relaxation reminders for arriving home, being at home in the afternoon or evening, and other quiet-at-home moments. |
| `R_LONG_DRIVE_REST` | You have been driving for quite a while. Find a suitable place to take a break. | 1 | Keeps the strong reminder to rest after a long period of driving. |
| `R_MEAL_BREAK` | It is about time to eat something. A short break with food will feel better too. | 2 | Merges meal reminders for lunchtime and midday outings from the office. |
| `R_NOISY_ENV_STAY_SAFE` | It is a bit noisy around you. Keep an eye on safety and your important belongings. | 2 | Merges safety reminders for crowded noisy places and noisy late-night environments. |
| `R_OPEN_CURTAINS_OR_LIGHT` | It has been dark for quite a while. Consider turning on a light or opening the curtains to reset your state. | 1 | Keeps the reminder to adjust the environment when it has been dark for too long. |
| `R_OUTDOOR_HYDRATE_AND_REST` | Do not forget to hydrate during outdoor activity. If you are tired, find a place to rest for a bit. | 2 | Merges hydration and rest reminders across outdoor running and long walking scenarios. |
| `R_PLAN_DAY_OVER_COFFEE` | Take a few minutes to think about the most important things for today and ease into the day. | 2 | Consolidates "having coffee," "starting the day," and "thinking through priorities" into one light start-of-day reminder. |
| `R_POST_WORKOUT_RECOVERY` | After exercise, remember to stretch and hydrate, and give your body some time to recover. | 4 | Merges recovery reminders across post-gym stretching, recovery after long training, and hydrating after exercise. |
| `R_PREPARE_FOR_MEETING` | The meeting is about to start. A quick preparation now will help you feel more ready. | 1 | Keeps the preparation reminder for right before a meeting starts. |
| `R_SLEEP_SOON_AND_CHARGE` | It is already quite late. Try to get ready to rest, and do not forget to take care of both yourself and your phone. | 3 | Merges "it is time to rest" reminders across late night at home, returning home late, and lying in bed using the phone late at night. |
| `R_WAITING_TIME_USEFUL` | You can make use of this waiting time to check your plans or take care of a small task. | 1 | Merges reminders that waiting time in education-related scenarios can be used to check plans or handle small tasks. |
| `R_WEEKEND_OR_MORNING_GREETING` | You can ease into today comfortably. No need to rush. | 2 | Merges light "ease into the day" reminders for waking up and weekend mornings. |
| `R_WORK_BREAK_AND_STRETCH` | You have been at it for a while. Get up and move a bit, and do not forget water and a short eye break. | 7 | Merges reminders to rest, hydrate, and move around across office work, work from home, prolonged sitting, and long focus sessions. |
| `R_WORK_START_SETTLE_IN` | Settle into a steady rhythm first, then the rest of the work will feel smoother. | 2 | Covers reminders about settling back into work after arriving at the office or returning after lunch. |
| `R_WORKOUT_PREP` | Warm up and drink some water before you start training. It will feel steadier. | 3 | Merges prep reminders such as warming up for morning exercise, getting ready to resume after a gym rest, and hydrating before heading to the gym. |
| `R_WRAP_UP_AND_REST_AFTER_WORK` | It is getting late. Try to wrap up what you are doing and leave yourself some room to rest. | 2 | Merges reminders to wrap up when staying late at the office or working overtime late at night. |

## Global Operation Space

| global O actionId | message | merged legacy count | merge note | operation |
| --- | --- | --- | --- | --- |
| `O_ENABLE_VIBRATION` | Enable vibration alerts. | 3 | Merges vibration-alert operations in noisy environments. | `enable_vibration` |
| `O_INCREASE_NOTIFICATION_VOLUME` | Increase the notification volume. | 3 | Merges `increase_volume` and `increase_volume_to_max`; the exact level is defined by parameters in JSON. | `increase_volume` |
| `O_OPEN_NOTES_QUICK_ENTRY` | Open the quick note entry. | 1 | Keeps the operation that opens the quick note entry during meetings. | `open_notes_quick_entry` |
| `O_PROMPT_ADD_FREQUENT_PLACE` | Prompt the user to add the current place as a frequent place. | 1 | Keeps the operation for prompting the user to add a frequent place after staying too long at an unknown location. | `prompt_add_frequent_place` |
| `O_SHOW_BOOKING_DETAILS` | Show the details of this booking or order. | 3 | Merges booking/order detail viewing actions for hotels, movies, hospitals, and similar cases. | `show_booking_details` |
| `O_SHOW_COMMUTE_TRAFFIC` | Show traffic and the estimated arrival time. | 3 | Merges commute traffic viewing actions for going to work, leaving work, and after leaving the office. | `show_commute_traffic` |
| `O_SHOW_MEETING_DETAILS` | Show the meeting time, location, and related information. | 2 | Merges meeting-info viewing actions for upcoming meetings and offsite meetings. | `show_meeting_details` |
| `O_SHOW_NEARBY_OPTIONS` | Show suitable nearby options. | 5 | Merges nearby search actions such as nearby places, nearby restaurants, and nearby service areas. | `show_nearby_places` |
| `O_SHOW_PAYMENT_QR` | Get the payment QR ready first. | 6 | Merges payment QR operations across dining, shopping, cafés, movies, and similar scenarios. | `show_payment_qr` |
| `O_SHOW_PICKUP_DETAILS` | Show the pickup details. | 1 | Keeps the pickup information viewing operation. | `show_pickup_details` |
| `O_SHOW_SCHEDULE` | Show the schedule. | 17 | Merges viewing today’s schedule, tomorrow’s plan, and schedule overview. | `show_schedule` |
| `O_SHOW_TODAY_TODO` | Show today's to-do items. | 14 | Merges all scenarios that show today's to-do list into one shared operation. | `show_today_todo_list` |
| `O_SHOW_TRAVEL_INFO` | Show the key details of the current trip. | 5 | Merges trip detail viewing actions across transit hubs, trains, flights, rideshares, and offsite meetings. | `show_travel_information` |
| `O_SHOW_WALKING_ROUTE` | Show a nearby walking route or return walking route. | 3 | Merges walking-route actions for park walks, after-meal walks, and weekend walks. | `show_walking_route` |
| `O_SHOW_WEATHER` | Show the weather. | 7 | Merges current-weather and today-weather viewing actions. | `show_weather` |
| `O_START_ACTIVITY_TRACKING` | Start tracking this activity. | 7 | Merges tracking actions for morning exercise, gym workouts, running, cycling, walking, and similar activities. | `start_activity_tracking` |
| `O_START_CONTEXT_TIMER` | Start an appropriate timer. | 7 | Merges general timer, rest timer, and recovery timer operations. | `start_context_timer` |
| `O_TURN_ON_DND` | Turn on Do Not Disturb. | 3 | Merges Do Not Disturb operations across meetings and pre-sleep scenarios. | `turn_on_dnd` |
| `O_TURN_ON_EYE_COMFORT_MODE` | Turn on eye comfort mode. | 8 | Merges all eye-comfort-mode related operations into one shared action. | `turn_on_eye_comfort_mode` |
| `O_TURN_ON_SILENT_MODE` | Turn on silent mode. | 2 | Merges silent mode operations in schools and quiet cafés. | `turn_on_silent_mode` |

## Scenario Defaults

| scenarioId | scenarioNameEn | default R/O actionIds | default app categories |
| --- | --- | --- | --- |
| `ARRIVE_OFFICE` | Arrive at office | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_PLAN_DAY_OVER_COFFEE` | `productivity`<br>`news`<br>`music` |
| `OFFICE_LUNCH_OUT` | Office lunch outing | `O_SHOW_PAYMENT_QR`<br>`R_MEAL_BREAK` | `shopping`<br>`social`<br>`navigation` |
| `OFFICE_WORKING` | Working at the office | `O_SHOW_TODAY_TODO`<br>`O_SHOW_SCHEDULE`<br>`R_WORK_BREAK_AND_STRETCH` | `productivity`<br>`music`<br>`news` |
| `LEAVE_OFFICE` | Leave office | `O_SHOW_COMMUTE_TRAFFIC`<br>`O_SHOW_WEATHER`<br>`R_CLOCK_OUT_BEFORE_LEAVING` | `navigation`<br>`music`<br>`social` |
| `WEEKDAY_HOME_DAY` | Weekday at home during the day | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_WORK_BREAK_AND_STRETCH` | `productivity`<br>`reading`<br>`music` |
| `ARRIVE_TRANSIT_HUB` | Arrive at transit hub | `O_SHOW_TRAVEL_INFO`<br>`R_CHECK_TRIP_OR_BOOKING_INFO` | `navigation`<br>`productivity`<br>`news` |
| `LATE_NIGHT_OVERTIME` | Late-night overtime | `R_WRAP_UP_AND_REST_AFTER_WORK`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `navigation`<br>`productivity`<br>`music` |
| `WEEKEND_OVERTIME` | Weekend overtime | `R_WORK_BREAK_AND_STRETCH`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `productivity`<br>`music`<br>`news` |
| `EVENING_AT_OFFICE` | Evening at the office | `O_SHOW_COMMUTE_TRAFFIC`<br>`O_SHOW_TODAY_TODO`<br>`R_WRAP_UP_AND_REST_AFTER_WORK` | `social`<br>`navigation`<br>`productivity` |
| `MORNING_EXERCISE` | Morning exercise | `O_START_ACTIVITY_TRACKING`<br>`O_START_CONTEXT_TIMER`<br>`R_WORKOUT_PREP` | `health`<br>`music`<br>`productivity` |
| `SHOPPING` | Shopping | `O_SHOW_PAYMENT_QR`<br>`R_ENJOY_LEISURE_MOMENT` | `shopping`<br>`social`<br>`navigation` |
| `HOME_EVENING` | Evening at home | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_HOME_RELAX_AND_RESET` | `entertainment`<br>`music`<br>`reading` |
| `ARRIVE_DINING` | Arrive at a restaurant/café | `O_SHOW_PAYMENT_QR`<br>`R_ENJOY_LEISURE_MOMENT` | `shopping`<br>`social`<br>`navigation` |
| `HOME_LATE_NIGHT` | Late night at home | `R_SLEEP_SOON_AND_CHARGE`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`O_SHOW_SCHEDULE` | `music`<br>`reading`<br>`health` |
| `WEEKEND_MORNING` | Weekend morning | `O_SHOW_WEATHER`<br>`O_SHOW_SCHEDULE`<br>`R_WEEKEND_OR_MORNING_GREETING` | `news`<br>`entertainment`<br>`game` |
| `WEEKEND_OUTING` | Weekend outing | `O_SHOW_WEATHER`<br>`O_SHOW_NEARBY_OPTIONS`<br>`R_ENJOY_LEISURE_MOMENT` | `navigation`<br>`social`<br>`entertainment` |
| `LATE_RETURN_HOME` | Late return home | `R_SLEEP_SOON_AND_CHARGE`<br>`O_TURN_ON_DND` | `music`<br>`health`<br>`reading` |
| `HOME_AFTERNOON` | Afternoon at home | `O_SHOW_TODAY_TODO`<br>`O_SHOW_SCHEDULE`<br>`R_HOME_RELAX_AND_RESET` | `reading`<br>`entertainment`<br>`music` |
| `MORNING_COFFEE` | Morning coffee | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_PLAN_DAY_OVER_COFFEE` | `productivity`<br>`news`<br>`music` |
| `GYM_WORKOUT` | Gym workout | `O_START_ACTIVITY_TRACKING`<br>`O_START_CONTEXT_TIMER`<br>`R_POST_WORKOUT_RECOVERY` | `health`<br>`music`<br>`productivity` |
| `GYM_REST` | Gym rest | `O_START_CONTEXT_TIMER`<br>`R_WORKOUT_PREP` | `health`<br>`music`<br>`productivity` |
| `GYM_LONG_STAY` | Long stay at the gym | `R_POST_WORKOUT_RECOVERY`<br>`O_START_CONTEXT_TIMER` | `health`<br>`music`<br>`productivity` |
| `OUTDOOR_RUNNING` | Outdoor running | `O_START_ACTIVITY_TRACKING`<br>`O_START_CONTEXT_TIMER`<br>`R_OUTDOOR_HYDRATE_AND_REST` | `health`<br>`music`<br>`navigation` |
| `WEEKEND_OUTDOOR_WALK` | Weekend outdoor walk | `O_SHOW_WALKING_ROUTE`<br>`O_SHOW_WEATHER`<br>`R_ENJOY_LEISURE_MOMENT` | `navigation`<br>`music`<br>`health` |
| `CYCLING` | Cycling | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_WEATHER`<br>`R_COMMUTE_STEADY_AND_SAFE` | `health`<br>`navigation`<br>`music` |
| `COMMUTE_MORNING` | Morning commute arrival at office | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_WORK_START_SETTLE_IN` | `navigation`<br>`music`<br>`news` |
| `COMMUTE_EVENING` | Arrive home after work | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_HOME_RELAX_AND_RESET` | `music`<br>`social`<br>`entertainment` |
| `HOME_TO_GYM` | Arrive at the gym from home | `O_START_ACTIVITY_TRACKING`<br>`O_START_CONTEXT_TIMER`<br>`R_WORKOUT_PREP` | `health`<br>`music`<br>`navigation` |
| `WAKE_UP` | Wake up at home | `O_SHOW_SCHEDULE`<br>`O_SHOW_WEATHER`<br>`R_WEEKEND_OR_MORNING_GREETING` | `news`<br>`productivity`<br>`music` |
| `PARK_WALK` | Short walk in the park | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_WALKING_ROUTE`<br>`R_ENJOY_LEISURE_MOMENT` | `navigation`<br>`health`<br>`music` |
| `LONG_DRIVE` | Long drive | `R_LONG_DRIVE_REST`<br>`O_SHOW_NEARBY_OPTIONS` | `navigation`<br>`music`<br>`news` |
| `CAFE_STAY` | Staying at a café | `O_SHOW_PAYMENT_QR`<br>`O_SHOW_SCHEDULE`<br>`R_ENJOY_LEISURE_MOMENT` | `reading`<br>`productivity`<br>`social` |
| `QUIET_HOME` | Quiet time at home | `R_HOME_RELAX_AND_RESET`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `reading`<br>`music`<br>`game` |
| `NOISY_GATHERING` | Noisy gathering | `O_ENABLE_VIBRATION`<br>`O_INCREASE_NOTIFICATION_VOLUME`<br>`R_NOISY_ENV_STAY_SAFE` | `social`<br>`music`<br>`navigation` |
| `LONG_OUTDOOR_WALK` | Long outdoor walk | `R_OUTDOOR_HYDRATE_AND_REST`<br>`O_SHOW_NEARBY_OPTIONS` | `health`<br>`navigation`<br>`music` |
| `LATE_NIGHT_NOISY` | Noisy place late at night | `O_ENABLE_VIBRATION`<br>`O_INCREASE_NOTIFICATION_VOLUME`<br>`R_NOISY_ENV_STAY_SAFE` | `navigation`<br>`social`<br>`music` |
| `DELIVERY_AT_OFFICE` | Delivery at the office | `O_SHOW_PICKUP_DETAILS`<br>`R_DELIVERY_PICKUP_REMINDER` | `shopping`<br>`productivity`<br>`social` |
| `EDUCATION_WAITING` | Waiting during class/tutoring | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_WAITING_TIME_USEFUL` | `reading`<br>`productivity`<br>`game` |
| `EDUCATION_LONG_SIT` | Long sitting during class/tutoring | `R_WORK_BREAK_AND_STRETCH`<br>`O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO` | `reading`<br>`productivity`<br>`health` |
| `HOME_LUNCH` | Lunch at home | `R_MEAL_BREAK`<br>`O_SHOW_NEARBY_OPTIONS` | `shopping`<br>`social`<br>`entertainment` |
| `AFTER_LUNCH_WALK` | After-meal walk | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_WALKING_ROUTE`<br>`R_AFTER_MEAL_WALK` | `health`<br>`navigation`<br>`music` |
| `MEETING_UPCOMING` | Meeting starting soon | `O_SHOW_MEETING_DETAILS`<br>`O_TURN_ON_DND`<br>`R_PREPARE_FOR_MEETING` | `productivity`<br>`navigation`<br>`social` |
| `IN_MEETING` | In a meeting | `O_TURN_ON_DND`<br>`O_OPEN_NOTES_QUICK_ENTRY`<br>`R_CAPTURE_MEETING_FOLLOW_UPS` | `productivity`<br>`social`<br>`reading` |
| `REMOTE_MEETING` | Leaving for an offsite meeting | `O_SHOW_TRAVEL_INFO`<br>`O_SHOW_MEETING_DETAILS`<br>`R_DEPART_EARLY_FOR_OFFSITE` | `navigation`<br>`productivity`<br>`social` |
| `NO_MEETINGS` | No meetings today | `O_SHOW_TODAY_TODO`<br>`R_DEEP_WORK_WINDOW` | `productivity`<br>`music`<br>`reading` |
| `CAL_HAS_EVENTS` | Today's calendar reminder | `O_SHOW_SCHEDULE`<br>`R_CALENDAR_QUICK_LOOK` | `productivity`<br>`navigation`<br>`news` |
| `RETURN_OFFICE_AFTER_LUNCH` | Return to the office after lunch | `O_SHOW_TODAY_TODO`<br>`O_SHOW_SCHEDULE`<br>`R_WORK_START_SETTLE_IN` | `productivity`<br>`music`<br>`news` |
| `HOME_AFTER_GYM` | Home after workout | `R_POST_WORKOUT_RECOVERY`<br>`O_START_CONTEXT_TIMER` | `health`<br>`music`<br>`entertainment` |
| `COMMUTE_FROM_HOME` | Commuting from home | `O_SHOW_COMMUTE_TRAFFIC`<br>`R_COMMUTE_STEADY_AND_SAFE` | `navigation`<br>`music`<br>`news` |
| `OFFICE_TO_CAFE` | From office to café | `O_SHOW_PAYMENT_QR`<br>`R_ENJOY_LEISURE_MOMENT` | `social`<br>`shopping`<br>`reading` |
| `LATE_NIGHT_PHONE` | Late-night phone use while lying down | `R_SLEEP_SOON_AND_CHARGE`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `health`<br>`reading`<br>`music` |
| `OFFICE_FOCUS_LONG` | Long focused office session | `R_WORK_BREAK_AND_STRETCH`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `productivity`<br>`music`<br>`health` |
| `HOME_DARK_LONG` | Home too dark for too long during the day | `R_OPEN_CURTAINS_OR_LIGHT`<br>`O_SHOW_WEATHER`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `health`<br>`reading`<br>`music` |
| `UNKNOWN_LONG_STAY` | Long stay at an unknown location | `O_PROMPT_ADD_FREQUENT_PLACE`<br>`O_SHOW_NEARBY_OPTIONS` | `navigation`<br>`social`<br>`shopping` |
| `OFFICE_LONG_SESSION` | Extra-long office session | `R_WORK_BREAK_AND_STRETCH`<br>`O_SHOW_TODAY_TODO` | `productivity`<br>`health`<br>`music` |
| `HOME_EVENING_DARK` | Dark evening at home | `O_TURN_ON_EYE_COMFORT_MODE`<br>`R_HOME_RELAX_AND_RESET` | `reading`<br>`entertainment`<br>`music` |
| `HOME_EVENING_NOISY` | Noisy evening at home | `O_INCREASE_NOTIFICATION_VOLUME`<br>`O_ENABLE_VIBRATION`<br>`R_HOME_RELAX_AND_RESET` | `social`<br>`music`<br>`entertainment` |
| `SCHOOL_QUIET` | Quiet period at school | `O_TURN_ON_SILENT_MODE`<br>`R_DEEP_WORK_WINDOW` | `reading`<br>`productivity`<br>`music` |
| `CAFE_QUIET` | Quiet café | `O_TURN_ON_SILENT_MODE`<br>`O_SHOW_SCHEDULE`<br>`R_DEEP_WORK_WINDOW` | `reading`<br>`productivity`<br>`music` |
| `TRAIN_DEPARTURE` | High-speed rail trip reminder | `O_SHOW_TRAVEL_INFO`<br>`R_CHECK_TRIP_OR_BOOKING_INFO` | `navigation`<br>`news`<br>`productivity` |
| `FLIGHT_BOARDING` | Flight boarding reminder | `O_SHOW_TRAVEL_INFO`<br>`R_CHECK_TRIP_OR_BOOKING_INFO` | `navigation`<br>`news`<br>`productivity` |
| `HOTEL_CHECKIN` | Hotel check-in reminder | `O_SHOW_BOOKING_DETAILS`<br>`R_CHECK_TRIP_OR_BOOKING_INFO` | `navigation`<br>`social`<br>`productivity` |
| `MOVIE_TICKET` | Movie ticket reminder | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR`<br>`R_CHECK_TRIP_OR_BOOKING_INFO` | `entertainment`<br>`social`<br>`shopping` |
| `HOSPITAL_APPOINTMENT` | Hospital appointment reminder | `O_SHOW_BOOKING_DETAILS`<br>`R_CHECK_TRIP_OR_BOOKING_INFO` | `health`<br>`navigation`<br>`productivity` |
| `RIDESHARE_PICKUP` | Rideshare pickup reminder | `O_SHOW_TRAVEL_INFO`<br>`R_CHECK_TRIP_OR_BOOKING_INFO` | `navigation`<br>`social`<br>`music` |

# Context-Aware Data Specification (RL Model Input Specification)

> This document is the interface contract between the RL model team and the on-device rules-engine team. **Whenever code changes affect any enum values, fields, or scenario definitions below, this document must be updated at the same time.**
>
> Last updated: 2026-03-19

---

## 1. 7-Tuple Physical State (PhysicalState)

### 1.1 Time Slot (TimeSlot) — 9 values

| Value | Time Range | Meaning |
|---|---|---|
| `sleeping` | 0:00-5:30 | Late night / early morning |
| `dawn` | 5:30-7:00 | Dawn |
| `morning` | 7:00-9:00 | Morning |
| `forenoon` | 9:00-11:30 | Forenoon |
| `lunch` | 11:30-13:30 | Lunch |
| `afternoon` | 13:30-17:00 | Afternoon |
| `evening` | 17:00-19:30 | Evening |
| `night` | 19:30-22:00 | Night |
| `late_night` | 22:00-0:00 | Before midnight |

### 1.2 Location (LocationCategory) — 17 values

| Value | Meaning | Default Geofence Radius | Source |
|---|---|---|---|
| `home` | Home | 80m | Direct geofence passthrough |
| `work` | Office | 150m | Direct geofence passthrough |
| `restaurant` | Restaurant | 80m | Direct geofence passthrough |
| `cafe` | Cafe | 60m | Direct geofence passthrough |
| `gym` | Gym | 80m | Direct geofence passthrough |
| `metro` | Metro station | 100m | Direct geofence passthrough |
| `rail_station` | Rail station / train station | 200m | Direct geofence passthrough |
| `airport` | Airport | 500m | Direct geofence passthrough |
| `transit` | Transit hub (gas station / highway service area, etc.) | 100m | Direct geofence passthrough |
| `shopping` | Shopping mall | 120m | Direct geofence passthrough |
| `outdoor` | Outdoor (park / plaza) | 200m | Direct geofence passthrough / inferred: no WiFi + GPS + walking/cycling |
| `health` | Medical facility | 150m | Direct geofence passthrough |
| `social` | Social entertainment (KTV / bar) | 100m | Direct geofence passthrough |
| `education` | Education (school / tutoring / training) | 200m | Direct geofence passthrough |
| `custom` | User-defined | 80m | Direct geofence passthrough |
| `en_route` | En route | — | Inferred: no geofence + driving / transit / walking with WiFi |
| `unknown` | Unknown | — | No signal match |

### 1.3 Motion State (MotionCategory) — 7 values

| Value | Meaning | Decision Basis |
|---|---|---|
| `stationary` | Stationary | Acceleration + GPS speed < 0.5 m/s |
| `walking` | Walking | Acceleration + increasing step count |
| `running` | Running | Acceleration magnitude > 12 |
| `cycling` | Cycling | GPS 3-8 m/s + low cadence |
| `driving` | Driving | GPS > 5 m/s or driving inertia |
| `transit` | Public transit | BT/GPS mode |
| `unknown` | Unknown | Insufficient data |

### 1.4 Phone State (PhoneCategory) — 8 values

| Value | Meaning | Decision Basis |
|---|---|---|
| `in_use` | In use | Screen on + held in hand + normal sitting/standing |
| `holding_lying` | Using phone while lying down | Held in hand + lying posture (accelerometer detected) |
| `on_desk` | On desk, face up | Flat + screen facing up + ambient light present |
| `face_up` | Face up (in the dark) | Screen facing up + dark room |
| `in_pocket` | In pocket | Proximity = near + ambient light < 5 lux |
| `face_down` | Face down | Proximity = near + ambient light > 5 lux |
| `charging` | Charging | Charging (overrides other postures) |
| `unknown` | Unknown | No sensor data |

### 1.5 Light (LightCategory) — 4 values

| Value | Meaning | Threshold |
|---|---|---|
| `dark` | Very dark | < 5 lux |
| `dim` | Dim | 5-50 lux |
| `normal` | Normal | 50-500 lux |
| `bright` | Bright | > 500 lux |

### 1.6 Sound (SoundCategory) — 5 values

| Value | Meaning | Threshold |
|---|---|---|
| `silent` | Silent | < 25 dB |
| `quiet` | Quiet | 25-40 dB |
| `normal` | Normal | 40-55 dB |
| `noisy` | Noisy | > 55 dB |
| `unknown` | Unknown | Microphone disabled (default) |

### 1.7 Day Type (DayType) — 3 values

| Value | Meaning |
|---|---|
| `workday` | Workday (Monday-Friday and not a statutory holiday) |
| `weekend` | Weekend (Saturday/Sunday) |
| `holiday` | Holiday (statutory holiday or user-marked) |

---

## 2. State Definitions (StateCode)

> The rules engine and the RL model are peer systems for scenario inference: the rules engine relies on manually written conditions, while RL learns personalized patterns from historical data. **Their shared input is the StateChain**, an ordered time series of StateCode values. StateCode is encoded from the 7-Tuple by StateEncoder; it is mutually exclusive and collectively exhaustive, and exactly one state is active at any moment.

Source: `StateEncoder.ets`, with **64 states** in total (44 base states + 20 substates). See `docs/state_encoding_architecture.md` for the full encoding logic.

**Two-stage encoding**: the first stage maps Location × Motion × TimeSlot × DayType to 44 base states. The second stage checks whether Phone/Light/Sound signals on certain base states satisfy substate conditions for a sustained duration of at least 10 minutes; if so, the state is refined into a substate, otherwise the base state is kept.

### Full state list

**Home (7)**

| StateCode | Meaning | Encoding Condition |
|-----------|------|---------|
| `home_sleeping` | Sleeping at home | home + stationary + sleeping/late_night |
| `home_morning_workday` | Home on a workday morning | home + stationary/walking + dawn/morning + workday |
| `home_morning_rest` | Home on a rest-day morning | home + stationary/walking + dawn/morning + weekend/holiday |
| `home_daytime_workday` | Home on a workday daytime | home + stationary/walking + forenoon/lunch/afternoon + workday |
| `home_daytime_rest` | Home on a rest-day daytime | home + stationary/walking + forenoon/lunch/afternoon + weekend/holiday |
| `home_evening` | Home in the evening/night | home + stationary/walking + evening/night |
| `home_active` | Active at home | home + running/cycling |

**Office (6)**

| StateCode | Meaning | Encoding Condition |
|-----------|------|---------|
| `office_arriving` | Arrive at office | work + dawn/morning + workday |
| `office_lunch_break` | Office lunch break | work + lunch + workday |
| `office_working` | Working in office | work + forenoon/afternoon + workday |
| `office_overtime` | Overtime | work + evening/night + workday |
| `office_late_overtime` | Late-night overtime | work + sleeping/late_night + workday |
| `office_rest_day` | In office on a rest day | work + weekend/holiday |

**Commuting (10)**

| StateCode | Meaning | Encoding Condition |
|-----------|------|---------|
| `commuting_walk_out` | Walk commute to work | walking + en_route + dawn/morning + workday |
| `commuting_walk_home` | Walk commute home | walking + en_route + evening/night + workday |
| `commuting_cycle_out` | Cycle commute to work | cycling + dawn/morning + workday |
| `commuting_cycle_home` | Cycle commute home | cycling + evening/night + workday |
| `commuting_drive_out` | Drive commute to work | driving + dawn/morning + workday |
| `commuting_drive_home` | Drive commute home | driving + evening/night + workday |
| `commuting_transit_out` | Transit commute to work | transit + dawn/morning + workday |
| `commuting_transit_home` | Transit commute home | transit + evening/night + workday |
| `driving` | Non-commute driving | driving (fallback) |
| `in_transit` | Non-commute transit | transit (fallback) |

**Transit locations (4)**

| StateCode | Meaning | Encoding Condition |
|-----------|------|---------|
| `at_metro` | Metro station | location=metro |
| `at_rail_station` | Rail station / train station | location=rail_station |
| `at_airport` | Airport | location=airport |
| `at_transit_hub` | Gas station / highway service area | location=transit |

**Outdoor (4)**

| StateCode | Meaning | Encoding Condition |
|-----------|------|---------|
| `outdoor_walking` | Outdoor walking | outdoor + walking |
| `outdoor_running` | Running | outdoor/unknown + running |
| `outdoor_cycling` | Cycling | outdoor/unknown + cycling (non-commute) |
| `outdoor_resting` | Resting outdoors | outdoor + stationary |

**Dining (3)**

| StateCode | Meaning | Encoding Condition |
|-----------|------|---------|
| `at_restaurant_lunch` | Restaurant lunch | restaurant + lunch |
| `at_restaurant_dinner` | Restaurant dinner | restaurant + evening/night |
| `at_restaurant_other` | Restaurant at other times | restaurant (fallback) |

**Other known locations (8)**

| StateCode | Meaning | Encoding Condition |
|-----------|------|---------|
| `at_cafe` | Cafe | location=cafe |
| `at_gym_exercising` | Exercising at the gym | gym + running/cycling |
| `at_gym` | At gym, stationary/walking | gym + stationary/walking |
| `at_shopping` | Shopping mall | location=shopping |
| `at_health` | Hospital / clinic | location=health |
| `at_social` | KTV/bar | location=social |
| `at_education` | School / tutoring | location=education |
| `at_custom` | User-defined location | location=custom |

**Fallback (2)**

| StateCode | Meaning | Encoding Condition |
|-----------|------|---------|
| `stationary_unknown` | Stationary in unknown location / no data | stationary or motion=unknown (final fallback) |
| `walking_unknown` | Walking in unknown location | walking (non-commute, non-outdoor) |

**Base states subtotal: 44**

---

### Substates (refined by Phone/Light/Sound, +20)

After a specific base state is matched, the Phone/Light/Sound signals are checked; if the conditions are satisfied, the state is refined into a substate. Selection rule: only signal values that **usually last at least 10 minutes** are included (excluding short-lived behaviors such as `phone: in_use`).

**Home substates (+10)**

| StateCode | Base State | Trigger Signal | Meaning |
|-----------|---------|---------|------|
| `home_sleeping_lying` | home_sleeping | phone=holding_lying | Using phone while lying down and not asleep |
| `home_morning_workday_lying` | home_morning_workday | phone=holding_lying | Staying in bed on a workday morning |
| `home_morning_rest_lying` | home_morning_rest | phone=holding_lying | Staying in bed on a rest-day morning |
| `home_daytime_workday_dark` | home_daytime_workday | light=dark | Lights off during workday daytime (nap / illness) |
| `home_daytime_workday_lying` | home_daytime_workday | phone=holding_lying | Lying down during workday daytime |
| `home_daytime_rest_dark` | home_daytime_rest | light=dark | Lights off during rest-day daytime (nap) |
| `home_daytime_rest_lying` | home_daytime_rest | phone=holding_lying | Lying down during rest-day daytime |
| `home_evening_dark` | home_evening | light=dark | Lights off in the evening (movie / meditation) |
| `home_evening_lying` | home_evening | phone=holding_lying | Lying down in the evening |
| `home_evening_noisy` | home_evening | sound=noisy | Noisy evening (party / cooking) |

**Office substates (+2)**

| StateCode | Base State | Trigger Signal | Meaning |
|-----------|---------|---------|------|
| `office_working_focused` | office_working | phone=face_down | Phone face down, focused working |
| `office_working_noisy` | office_working | sound=noisy | In a meeting / discussion |

**Cafe substates (+1)**

| StateCode | Base State | Trigger Signal | Meaning |
|-----------|---------|---------|------|
| `at_cafe_quiet` | at_cafe | sound=quiet | Quiet cafe (good for work / reading) |

**Education-location substates (+2)**

| StateCode | Base State | Trigger Signal | Meaning |
|-----------|---------|---------|------|
| `at_education_class` | at_education | sound=quiet | In class |
| `at_education_break` | at_education | sound=noisy | Between classes / after class |

**Medical-location substates (+1)**

| StateCode | Base State | Trigger Signal | Meaning |
|-----------|---------|---------|------|
| `at_health_inpatient` | at_health | phone=holding_lying | Hospitalized / bedridden |

**Unknown-location substates (+4)**

| StateCode | Base State | Trigger Signal | Meaning |
|-----------|---------|---------|------|
| `unknown_noisy` | stationary_unknown | sound=noisy | Noisy unknown location |
| `unknown_dark` | stationary_unknown | light=dark | Very dark unknown location |
| `unknown_settled` | stationary_unknown | phone=charging | Charging in unknown location (settled in) |
| `unknown_lying` | stationary_unknown | phone=holding_lying | Lying down in unknown location |

**Total: 64 StateCode values (44 base states + 20 substates)**

---

## 3. DataTray Extended Fields (ContextSnapshot)

### 3.1 Basic Context

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `hour` | string | "0"-"23" | Current hour |
| `timeOfDay` | string | TimeSlot enum value | Current time slot (see 1.1) |
| `dayOfWeek` | string | "0"-"6" | Day of week (0=Sunday) |
| `isWeekend` | string | "true"/"false" | Whether it is the weekend |
| `batteryLevel` | string | "0"-"100" | Battery percentage |
| `isCharging` | string | "true"/"false" | Whether charging |
| `networkType` | string | wifi/cellular/none | Network type |
| `sensorTier` | string | "low"/"normal"/"high" | Sensor collection tier |

### 3.2 Location and Geofences

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `mergedGeofenceCategory` | string | PlaceCategory Value | Merged geofence category (WiFi first, GPS fallback) |
| `geofence` | string | Geofence ID | Current geofence |
| `wifiSsid` | string | SSID name | Current WiFi |
| `wifiGeofence` | string | Geofence ID | Geofence matched by WiFi (cleared on disconnect) |
| `wifiLost` | string | "true"/"false" | Whether a known WiFi was just lost |
| `wifiLostCategory` | string | PlaceCategory Value | Geofence category corresponding to lost WiFi |
| `wifiLostWork` | string | "true"/"false" | Whether office WiFi was left |
| `latitude` | string | Numeric value | Latitude |
| `longitude` | string | Numeric value | Longitude |
| `cellId` | string | Numeric value | Cell tower ID |
| `fusionGeofence` | string | Geofence ID | Fused-location geofence result |
| `fusionConfidence` | string | Numeric value 0-1 | Fused-location confidence |
| `fusionSource` | string | wifi/gps/bt/cell | Fused-location source |

### 3.3 Motion and Activity

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `motionState` | string | stationary/walking/running/driving/transit | Motion State |
| `prevMotionState` | string | Same as above | Previous motion state |
| `activityState` | string | sitting/sleeping/standing/active | Current activity state |
| `prevActivityState` | string | Same as above | Previous activity state |
| `activityDuration` | string | Numeric value (seconds) | Current activity duration |
| `stationaryCumulativeMin` | string | Numeric value (minutes) | Cumulative stationary time within a 60-minute rolling window |
| `stepCount` | string | Numeric value | Step count |
| `step_count_today` | string | Numeric value | Today's cumulative step count |
| `speed` | string | Numeric value (km/h) | Estimated speed |
| `transportMode` | string | walking/running/cycling/driving/transit/stationary | Transport mode |
| `isSleeping` | string | "true"/"false" | Whether sleeping |
| `isLyingDown` | string | "true"/"false" | Whether device is lying flat |

### 3.4 Calendar (CalendarPlugin)

Source: `CalendarPlugin.ets`. Reads calendar events created by this app through CalendarKit, refreshes the cache every 5 minutes, and looks ahead 36 hours.

> Limitation: HarmonyOS NEXT can only read calendar events created by this app; third-party calendars (such as WeLink) are not accessible. External calendar events are injected through `injectEvents()`.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `cal_eventCount` | string | Numeric value | Total number of future events (after the current time) |
| `cal_hasUpcoming` | string | "true"/"false" | Whether there is an upcoming event |
| `cal_nextTitle` | string | Text | Next event title |
| `cal_nextMinutes` | string | Numeric value | Minutes until next event (rounded) |
| `cal_nextLocation` | string | Text | Next event location (optional) |
| `cal_inMeeting` | string | "true"/"false" | Whether currently in a meeting (`startTime ≤ now < endTime`) |
| `cal_currentTitle` | string | Text | Title of current ongoing event (only when `cal_inMeeting=true`) |

### 3.5 Delivery / Parcel (SmsEntityPlugin)

Source: `SmsEntityPlugin.ets`. Scans recent SMS messages and uses a UIE NER model (C++ NAPI) to extract parcel-related entities; if the model is unavailable, it falls back to regular expressions. Rescans every 5 minutes.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `sms_delivery_pending` | string | "true"/"false" | Whether there is a parcel pending pickup (within a 24h window) |
| `sms_delivery_courier` | string | Text | Courier name (SF, ZTO, YTO, Yunda, STO, J&T, JD, Postal, EMS, etc.) |
| `sms_delivery_code` | string | Text | Pickup code (alphanumeric, 2-12 characters) |
| `sms_delivery_location` | string | Text | Pickup location (parcel station / locker / pickup point, etc.) |
| `sms_delivery_time` | string | Numeric value (Unix ms) | Parcel SMS timestamp |
| `sms_delivery_count` | string | Numeric value | Number of parcel SMS messages within 24h |

### 3.6 Travel (TravelInfoService)

Source: `TravelInfoService.ets`, which parses flight numbers and train numbers from calendar events; `ContextAwarenessService` writes the results into DataTray. Looks ahead 24 hours and refreshes every 30 minutes.

> Flights and trains share the same set of DataTray fields (flight takes precedence, then train). The `flightNumber` field may also store a train number.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `hasUpcomingFlight` | string | "true"/"false" | Whether there is an upcoming departing flight or train |
| `flightCountdownMin` | string | Numeric value (may be negative) | Minutes until takeoff/departure (negative = already departed) |
| `flightArrivalMin` | string | Numeric value (may be negative) | Minutes until landing/arrival (inferred from calendar `endTime`) |
| `flightNumber` | string | Text | Flight number (e.g. `CA1234`) or train number (e.g. `G1234`) |

### 3.7 Screen State (ScreenStatePlugin) — already included in reporting

Source: `ScreenStatePlugin.ets`, detected through the screenLock + display API and refreshed on each polling cycle (30s).

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `screen_locked` | string | "true"/"false" | Whether screen is locked |
| `screen_on` | string | "true"/"false" | Whether screen is on |
| `user_active` | string | "true"/"false" | Whether user is active (unlocked + screen on) |

### 3.8 Audio and Media (MediaStatePlugin) — already included in reporting

Source: `MediaStatePlugin.ets`, which uses AudioKit to detect audio scenes and output devices.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `audio_scene` | string | default/ringing/phone_call/voice_chat | Current audio scene |
| `audio_device` | string | speaker/wired_headset/wired_headphones/bluetooth_a2dp/bluetooth_sco/usb_headset/earpiece | Current audio output device |
| `audio_inCall` | string | "true"/"false" | Whether currently in a call (`phone_call` or `voice_chat`) |
| `audio_ringing` | string | "true"/"false" | Whether the phone is ringing |
| `audio_headphones` | string | "true"/"false" | Whether headphones are connected (wired/Bluetooth/USB) |

### 3.9 Bluetooth Devices (BluetoothPlugin) — already included in reporting

Source: `BluetoothPlugin.ets`, which detects Bluetooth state and classifies devices by majorMinorClass into fixed (speakers), portable (headphones/wearables), and vehicle (in-car).

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `bt_enabled` | string | "true"/"false" | Whether Bluetooth is enabled |
| `bt_connected_count` | string | Numeric value | Number of connected devices |
| `bt_audio_connected` | string | "true"/"false" | Whether an audio device is connected |
| `bt_device_names` | string | Comma-separated text | Names of all connected devices |
| `bt_fixed_devices` | string | Comma-separated text | Names of fixed devices (speaker/set-top box) → can be linked to geofences |
| `bt_portable_devices` | string | Comma-separated text | Names of portable devices (headphones/wearables) |
| `bt_vehicle_devices` | string | Comma-separated text | Names of in-vehicle devices |
| `bt_in_vehicle` | string | "true"/"false" | Whether in-vehicle Bluetooth is connected |

### 3.10 Foreground App (ForegroundAppPlugin) — already included in reporting

Source: `ForegroundAppPlugin.ets`, which infers app category by matching the bundleName against a local category table.

> Limitation: third-party apps cannot read usage statistics of other apps (requires `system_basic` permission), so this is updated only when an app switch is detected.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `foreground_app` | string | bundleName | Current foreground app package name |
| `app_category` | string | social/office/entertainment/navigation/shopping/news/health/music/reading/game/other | App category (local mapping) |
| `app_usage_min` | string | Numeric value | Current app usage duration (minutes) |

### 3.11 App Usage (AppUsagePlugin) — already included in reporting

Source: `AppUsagePlugin.ets`, which detects usage state and session information for this app.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `app_isForeground` | string | "true"/"false" | Whether this app is in the foreground |
| `app_sessionMinutes` | string | Numeric value | Current session duration (minutes) |
| `app_sessionsToday` | string | Numeric value | Number of uses today |

### 3.12 Notifications (NotificationPlugin) — already included in reporting

Source: `NotificationPlugin.ets`, which detects this app's notification status.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `notif_enabled` | string | "true"/"false" | Whether notification permission is enabled |
| `notif_selfCount` | string | Numeric value | Number of active notifications from this app |
| `notif_lastSentMinutes` | string | Numeric value | Minutes since the last notification was sent |

### 3.13 Phone Posture (PosturePlugin) — already included in reporting

Source: `PosturePlugin.ets`, which uses the accelerometer to determine phone placement posture.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `phone_posture` | string | face_up/face_down/standing/tilted/in_pocket | Phone placement posture |
| `is_flat` | string | "true"/"false" | Whether placed horizontally |

### 3.14 Ambient Sound (AmbientSoundPlugin) — already included in reporting

Source: `AmbientSoundPlugin.ets`, which analyzes ambient noise through microphone sampling.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `noise_level` | string | Numeric value (dB) | Ambient noise level |
| `sound_scene` | string | quiet/normal/noisy/unknown | Sound scene |
| `has_voice` | string | "true"/"false" | Whether voice is detected |
| `volume_db` | string | Numeric value | Ambient volume (dB) |

### 3.15 WiFi Details (WifiPlugin) — already included in reporting

Source: updated on WiFi connection events.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `wifi_connected` | string | "true"/"false" | Whether WiFi is connected |
| `wifi_rssi` | string | Numeric value (dBm) | WiFi signal strength |
| `wifi_frequency` | string | Numeric value (MHz) | WiFi frequency (2.4G/5G) |

### 3.16 Wearable Device (WearablePlugin) — already included in reporting

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `device_connected` | string | "true"/"false" | Whether a wearable device is connected |
| `device_name` | string | Text | Wearable device name |

### 3.17 Health and Wearables

Source: reported by wearable devices through Health Kit.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `heart_rate` | string | Numeric value (bpm) | Real-time heart rate |
| `heart_rate_status` | string | resting/normal/elevated/high | Heart-rate status |
| `wearing_state` | string | "true"/"false" | Whether the wearable is being worn |

### 3.18 State-Chain Fields (StateChain)

Written to DataTray by `StateChain.ets` whenever the state changes, and synchronized into ContextSnapshot by `CAS.encodeStateAndAugment()` during each evaluation.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `state_current` | string | 64 StateCode values (see Section 2) | Current state code |
| `state_prev` | string | 64 StateCode values or an empty string | Previous different state code |
| `state_duration_sec` | string | Numeric value (seconds) | Number of seconds the current state has lasted |
| `state_chain_json` | string | JSON array | Most recent 10 state entries (head = latest), format: `[{"code":"...","startTime":...,"durationMs":...}]` |

### 3.19 7-Tuple Snapshot Fields (`ps_*`)


Computed by `PhysicalStateBuilder` during each evaluation cycle and written into DataTray for use by the rules engine.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `ps_time` | string | TimeSlot enum value (see 1.1) | Current time slot |
| `ps_location` | string | LocationCategory enum value (see 1.2) | Current location category |
| `ps_motion` | string | MotionCategory enum value (see 1.3) | Motion State |
| `ps_phone` | string | PhoneCategory enum value (see 1.4) | Phone State |
| `ps_light` | string | LightCategory enum value (see 1.5) | Light level |
| `ps_sound` | string | SoundCategory enum value (see 1.6) | Sound level |
| `ps_dayType` | string | DayType enum value (see 1.7) | Day Type |

### 3.20 Scenario Matching Information

Written into DataTray by `ScenarioMatcher` after each evaluation.

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `ps_scenario` | string | scenarioId | Current highest-priority scenario ID |
| `ps_scenarioCategory` | string | Scenario category | Scenario category |
| `ps_chainPosition` | string | Numeric value | Position in the scenario chain |
| `ps_scenarioConfidence` | string | Numeric value 0-1 | Scenario-match confidence |

---

## 4. RL Training Database Schema

> The following is the actual storage schema on the PostgreSQL server. The training samples required by the RL model are assembled by joining these 3 tables.

---

### 4.1 `signal_reports` — Signal snapshot timeline

**Purpose**: one full sensor snapshot every 2 minutes. This is the main source of state features.

**Extensibility**: `signals` is a flat JSONB key-value object. Adding or removing DataTray fields only changes what the client writes; **the table schema does not need to change**. Different client versions may upload different sets of `signals` fields, and the server can read the needed keys on demand.

**Schema**:

| Column | Type | Description |
|------|------|------|
| `id` | BIGSERIAL PK | Auto-increment primary key |
| `device_id` | TEXT NOT NULL | Device identifier |
| `timestamp` | BIGINT NOT NULL | Snapshot timestamp (ms epoch) |
| `signals` | JSONB NOT NULL | Flat key-value object containing all DataTray fields; the field set evolves with versions |
| `created_at` | TIMESTAMPTZ | Server write time |

Index: `(device_id, timestamp)`

**Example row (v1.0 client, base fields)**:

```json
{
  "id": 1001,
  "device_id": "dev_abc123",
  "timestamp": 1742350800000,
  "signals": {
    "ps_time": "morning",
    "ps_location": "home",
    "ps_motion": "walking",
    "ps_dayType": "workday",
    "batteryLevel": "90",
    "isCharging": "false",
    "networkType": "wifi",
    "hour": "7"
  }
}
```

**Example row (v2.0 client, with added StateCode + calendar + wearable fields)**:

```json
{
  "id": 2048,
  "device_id": "dev_xyz789",
  "timestamp": 1742351700000,
  "signals": {
    "ps_time": "morning",
    "ps_location": "home",
    "ps_motion": "walking",
    "ps_dayType": "workday",
    "ps_light": "normal",
    "ps_sound": "quiet",
    "ps_phone": "in_pocket",
    "state_current": "home_morning_workday_lying",
    "state_prev": "home_sleeping",
    "state_duration_sec": "900",
    "batteryLevel": "90",
    "isCharging": "false",
    "networkType": "wifi",
    "hour": "7",
    "cal_hasUpcoming": "true",
    "cal_nextMinutes": "45",
    "heart_rate": "68",
    "wearing_state": "true",
    "bt_in_vehicle": "false"
  }
}
```

> The two rows coexist in the same table. During RL training, keys are read only if they actually exist; missing keys are treated as null.

---

### 4.2 `state_chain_entries` — State chain (one row per state, append-only)

**Purpose**: each state is stored as its own row, similar to the design of `signal_reports`. INSERT when a state starts, and UPDATE `end_time` / `duration_ms` when the state ends. This provides complete temporal context for Claude personalization and RL state-sequence features. Adding or removing StateCode values affects only the value of the `state_code` field, not the table schema.

**Schema**:

| Column | Type | Description |
|------|------|------|
| `id` | BIGSERIAL PK | Auto-increment primary key |
| `device_id` | TEXT NOT NULL | Device identifier |
| `state_code` | TEXT NOT NULL | StateCode value, e.g. `"office_working"` |
| `start_time` | BIGINT NOT NULL | State start time (ms epoch) |
| `end_time` | BIGINT | State end time (NULL = current state is still active) |
| `duration_ms` | BIGINT | Duration (NULL until the state ends and is filled in) |
| `created_at` | TIMESTAMPTZ | Server write time |

Index: `(device_id, start_time DESC)`

**Example rows**:

```
id   device_id     state_code             start_time        end_time          duration_ms
1    dev_abc123    home_sleeping          1742325300000     1742346900000     21600000
2    dev_abc123    home_morning_workday   1742346900000     1742349900000     3000000
3    dev_abc123    commuting_transit_out  1742349900000     1742351700000     1800000
4    dev_abc123    office_working         1742351700000     NULL              NULL        ← Current state
```

**Join query** (find the StateCode corresponding to time T):

```sql
SELECT state_code
FROM state_chain_entries
WHERE device_id = 'dev_abc123'
  AND start_time <= 1742351700000
  AND (end_time IS NULL OR end_time > 1742351700000);
```

**Read the most recent 100 history rows**:

```sql
SELECT state_code, start_time, end_time, duration_ms
FROM state_chain_entries
WHERE device_id = 'dev_abc123'
ORDER BY start_time DESC
LIMIT 100;
```

---

### 4.3 `state_pairs` — Rule matches × suggestions × feedback

**Purpose**: whenever a rules-engine match succeeds, one pair is recorded for each "rule × suggestion" combination, and feedback is filled in after user interaction. This is the direct source of the RL reward signal.

**Schema**:

| Column | Type | Description |
|------|------|------|
| `id` | BIGSERIAL PK | Auto-increment primary key |
| `device_id` | TEXT NOT NULL | Device identifier |
| `match_timestamp` | BIGINT NOT NULL | Rule-match timestamp (ms epoch) |
| `context_window_start` | BIGINT NOT NULL | Precondition look-back start time (ms epoch) |
| `scenario_id` | TEXT NOT NULL | Scenario ID, e.g. `"COMMUTE_MORNING"` |
| `scenario_name` | TEXT | Scenario name, e.g. `"Morning commute arrive at work"` |
| `rule_id` | TEXT NOT NULL | Matched rule ID, e.g. `"sc_commute_morning_arrive"` |
| `confidence` | DOUBLE PRECISION | Match confidence 0–1 |
| `suggestion` | JSONB NOT NULL | `{ "actionId": "...", "actionType": "..." }` |
| `feedback` | JSONB | `{ "feedbackType": "thumbs_up\|thumbs_down\|no_feedback", "reward": 1.0\|-0.5\|0 }` |
| `created_at` | TIMESTAMPTZ | Server write time |

Unique index (for UPSERT): `(device_id, rule_id, match_timestamp, (suggestion->>'actionId'))`
Indexes: `(device_id, match_timestamp)`, `(scenario_id)`, `(rule_id)`

**Example rows** (3 rows produced in the same evaluation):

```json
[
  {
    "device_id": "dev_abc123",
    "match_timestamp": 1742351700000,
    "context_window_start": 1742344500000,
    "scenario_id": "COMMUTE_MORNING",
    "rule_id": "sc_commute_morning_arrive",
    "confidence": 0.95,
    "suggestion": { "actionId": "arrive_work_schedule", "actionType": "card" },
    "feedback": { "feedbackType": "thumbs_up", "reward": 1.0 }
  },
  {
    "device_id": "dev_abc123",
    "match_timestamp": 1742351700000,
    "context_window_start": 1742344500000,
    "scenario_id": "ARRIVE_OFFICE",
    "rule_id": "sc_02",
    "confidence": 0.90,
    "suggestion": { "actionId": "arrive_office_coffee", "actionType": "suggestion" },
    "feedback": { "feedbackType": "no_feedback", "reward": 0 }
  },
  {
    "device_id": "dev_abc123",
    "match_timestamp": 1742351700000,
    "context_window_start": 1742344500000,
    "scenario_id": "ARRIVE_OFFICE",
    "rule_id": "sc_02",
    "confidence": 0.90,
    "suggestion": { "actionId": "none", "actionType": "none" },
    "feedback": { "feedbackType": "no_feedback", "reward": 0 }
  }
]
```

---

### 4.4 RL Training Sample Assembly

A complete RL training sample is assembled by joining the 5 tables above:

```
┌─────────────────────────────────────────────────────────────────┐
│                      One RL training sample                             │
│                                                                 │
│  state_features          shown_actions      selected_action_id  │
│  ─────────────           ────────────       ──────────────────  │
│  From signal_reports      Same T±5s window      feedback =          │
│  .signals (most recent snapshot)      all state_pairs in the window  the row with `thumbs_up`    │
│                           aggregated         pair.suggestion    │
│                                                                 │
│  state_sequence          scenario_id        reward             │
│  ──────────────          ───────────        ──────             │
│  From state_chain_entries state_pairs        feedback.reward    │
│  ORDER BY start_time     .scenario_id       1.0 / -0.5 / 0    │
└─────────────────────────────────────────────────────────────────┘
```

**Assembly SQL example**:

```sql
-- Assemble the training sample for match T=1742351700000
WITH match AS (
  SELECT * FROM state_pairs
  WHERE device_id = 'dev_abc123'
    AND match_timestamp BETWEEN 1742351695000 AND 1742351705000
),
features AS (
  SELECT signals FROM signal_reports
  WHERE device_id = 'dev_abc123'
    AND timestamp <= 1742351700000
  ORDER BY timestamp DESC LIMIT 1
),
state_seq AS (
  SELECT json_agg(json_build_object(
    'code', state_code,
    'startTime', start_time,
    'endTime', end_time,
    'durationMs', duration_ms
  ) ORDER BY start_time DESC) AS chain
  FROM state_chain_entries
  WHERE device_id = 'dev_abc123'
  LIMIT 100
)
SELECT
  m.scenario_id,
  m.match_timestamp,
  f.signals          AS state_features,
  s.chain            AS state_sequence,
  json_agg(json_build_object(
    'actionId',    m.suggestion->>'actionId',
    'actionType',  m.suggestion->>'actionType',
    'ruleId',      m.rule_id,
    'confidence',  m.confidence,
    'reward',      (m.feedback->>'reward')::float
  ))                 AS shown_actions
FROM match m, features f, state_seq s
GROUP BY m.scenario_id, m.match_timestamp, f.signals, s.chain;
```

**Quick field-to-table mapping**:

| Sample Field | Source Table | How Obtained |
|---------|--------|------|
| `timestamp` | `state_pairs.match_timestamp` | Direct |
| `scenario_id` | `state_pairs.scenario_id` | Direct |
| `state_features` | `signal_reports.signals` | `timestamp <= T ORDER BY DESC LIMIT 1` |
| `state_sequence` | `state_chain_entries` | `WHERE device_id = X ORDER BY start_time DESC LIMIT 100` |
| `shown_actions` | `state_pairs` (same T window) | `match_timestamp BETWEEN T-5000 AND T+5000` |
| `selected_action` | `state_pairs.suggestion` | `WHERE feedback->>'feedbackType' = 'thumbs_up'` |
| `reward` | `state_pairs.feedback.reward` | `1.0 / -0.5 / 0` |

---

## 5. Cloud-Service Reporting Scope

### 5.1 Reporting Path

On-device data is uploaded to the cloud analytics server after ContextSnapshot is built in two stages:

1. **C++ `getSnapshot()`**: returns 19 core fields (`hour`, `dayOfWeek`, `isWeekend`, `batteryLevel`, `isCharging`, `networkType`, `wifiSsid`, `latitude`, `longitude`, `geofence`, `motionState`, `stepCount`, `step_count_today`, `speed`, `transportMode`, `activityState`, `activityDuration`, `isSleeping`, `isLyingDown`)
2. **ETS `augmentSnapshot()`**: supplements the full set of plugin fields from DataTray (all fields in Sections 3.7-3.19)

### 5.2 Design Principles

- **DataTray as the only intermediate layer**: all data must be stored in DataTray, and the engine, UI, and cloud reporting read only through DataTray.
- **Full reporting**: at the current stage, all plugin fields are included in reporting through `augmentSnapshot`, with no field filtering.
- **Future plan**: add a privacy-labeling mechanism (fields marked private will not be uploaded; unmarked fields will be uploaded in full).


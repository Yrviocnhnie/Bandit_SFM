'''
python generate_in_between_scenarios.py   --spec in_between_scenarios_spec.json   --output in_between_scenarios_1000_each.jsonl   --metadata metainfo/in_between_scenarios_1000_each_metadata.json   --samples-per-scenario 1000   --seed 7
'''
#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from typing import Any

TIME_SLOT_RANGES = {
    'sleeping': (0, 5*3600 + 30*60),
    'dawn': (5*3600 + 30*60, 7*3600),
    'morning': (7*3600, 9*3600),
    'forenoon': (9*3600, 11*3600 + 30*60),
    'lunch': (11*3600 + 30*60, 13*3600 + 30*60),
    'afternoon': (13*3600 + 30*60, 17*3600),
    'evening': (17*3600, 19*3600 + 30*60),
    'night': (19*3600 + 30*60, 22*3600),
    'late_night': (22*3600, 24*3600),
}

AGE_BUCKETS = ['0_17', '18_24', '25_34', '35_44', '45_54', '55_64', '65_plus', 'unknown']
SEX_VALUES = ['female', 'male', 'other', 'unknown']
APP_CATEGORIES = ['social', 'productivity', 'entertainment', 'navigation', 'shopping', 'news', 'health', 'music', 'reading', 'game']
SMS_FIELDS = ['sms_delivery_pending','sms_train_pending','sms_flight_pending','sms_hotel_pending','sms_movie_pending','sms_hospital_pending','sms_ride_pending']

STATE_PROFILES = {
    'office_arriving': {
        'ps_location': ['work'], 'ps_time': ['dawn','morning'], 'ps_dayType': ['workday'],
        'ps_motion': ['stationary','walking'], 'ps_phone': ['face_down','face_up','in_pocket','in_use','on_desk'],
        'ps_light': ['normal','bright'], 'ps_sound': ['quiet','normal'], 'networkType': ['wifi','cellular'],
        'isCharging': [0], 'activityState_by_motion': {'stationary':['sitting','standing'], 'walking':['standing','active']}
    },
    'office_working': {
        'ps_location': ['work'], 'ps_time': ['forenoon','afternoon'], 'ps_dayType': ['workday'],
        'ps_motion': ['stationary','walking'], 'ps_phone': ['face_down','face_up','in_pocket','in_use','on_desk'],
        'ps_light': ['normal','bright'], 'ps_sound': ['quiet','normal'], 'networkType': ['wifi','cellular'],
        'isCharging': [0], 'activityState_by_motion': {'stationary':['sitting','standing'], 'walking':['standing','active']}
    },
    'office_working_focused': {
        'ps_location': ['work'], 'ps_time': ['forenoon','afternoon'], 'ps_dayType': ['workday'],
        'ps_motion': ['stationary'], 'ps_phone': ['face_down'], 'ps_light': ['normal','bright'], 'ps_sound': ['quiet','normal'],
        'networkType': ['wifi','cellular'], 'isCharging': [0], 'activityState_by_motion': {'stationary':['sitting','standing']}
    },
    'office_overtime': {
        'ps_location': ['work'], 'ps_time': ['evening','night'], 'ps_dayType': ['workday'],
        'ps_motion': ['stationary','walking'], 'ps_phone': ['face_down','face_up','in_pocket','in_use','on_desk'],
        'ps_light': ['dim','normal'], 'ps_sound': ['quiet','normal'], 'networkType': ['wifi','cellular'],
        'isCharging': [0], 'activityState_by_motion': {'stationary':['sitting','standing'], 'walking':['standing','active']}
    },
    'office_late_overtime': {
        'ps_location': ['work'], 'ps_time': ['night','late_night'], 'ps_dayType': ['workday'],
        'ps_motion': ['stationary'], 'ps_phone': ['face_down','face_up','in_use','on_desk'],
        'ps_light': ['dim','normal'], 'ps_sound': ['quiet','normal'], 'networkType': ['wifi','cellular'],
        'isCharging': [0], 'activityState_by_motion': {'stationary':['sitting','standing']}
    },
    'at_cafe': {
        'ps_location': ['cafe'], 'ps_time': ['morning','forenoon','lunch','afternoon','evening','night'], 'ps_dayType': ['workday','weekend','holiday'],
        'ps_motion': ['stationary','walking'], 'ps_phone': ['face_up','in_pocket','in_use','on_desk'],
        'ps_light': ['dim','normal','bright'], 'ps_sound': ['quiet','normal'], 'networkType': ['wifi','cellular'],
        'isCharging': [0], 'activityState_by_motion': {'stationary':['sitting','standing'], 'walking':['standing','active']}
    },
    'at_cafe_quiet': {
        'ps_location': ['cafe'], 'ps_time': ['morning','forenoon','lunch','afternoon','evening','night'], 'ps_dayType': ['workday','weekend','holiday'],
        'ps_motion': ['stationary','walking'], 'ps_phone': ['face_up','in_pocket','in_use','on_desk'],
        'ps_light': ['dim','normal','bright'], 'ps_sound': ['quiet'], 'networkType': ['wifi','cellular'],
        'isCharging': [0], 'activityState_by_motion': {'stationary':['sitting','standing'], 'walking':['standing','active']}
    },
    'outdoor_walking': {
        'ps_location': ['outdoor'], 'ps_time': ['morning','forenoon','lunch','afternoon','evening','night'], 'ps_dayType': ['workday','weekend','holiday'],
        'ps_motion': ['walking'], 'ps_phone': ['face_up','in_pocket','in_use'], 'ps_light': ['normal','bright'], 'ps_sound': ['quiet','normal'],
        'networkType': ['cellular'], 'isCharging': [0], 'activityState_by_motion': {'walking':['standing','active']}
    },
    'home_morning_workday': {
        'ps_location': ['home'], 'ps_time': ['dawn','morning'], 'ps_dayType': ['workday'], 'ps_motion': ['stationary','walking'],
        'ps_phone': ['face_up','in_pocket','in_use','on_desk'], 'ps_light': ['dim','normal','bright'], 'ps_sound': ['quiet','normal'],
        'networkType': ['wifi','cellular'], 'isCharging': [0,1], 'activityState_by_motion': {'stationary':['sitting','standing'], 'walking':['standing','active']}
    },
    'home_daytime_workday': {
        'ps_location': ['home'], 'ps_time': ['forenoon','lunch','afternoon'], 'ps_dayType': ['workday'], 'ps_motion': ['stationary','walking'],
        'ps_phone': ['face_up','in_pocket','in_use','on_desk'], 'ps_light': ['normal','bright'], 'ps_sound': ['quiet','normal'],
        'networkType': ['wifi','cellular'], 'isCharging': [0,1], 'activityState_by_motion': {'stationary':['sitting','standing'], 'walking':['standing','active']}
    },
    'at_metro': {
        'ps_location': ['metro'], 'ps_time': ['dawn','morning','forenoon','lunch','afternoon','evening','night'], 'ps_dayType': ['workday','weekend','holiday'],
        'ps_motion': ['stationary'], 'ps_phone': ['face_up','in_pocket','in_use'], 'ps_light': ['dim','normal','bright'], 'ps_sound': ['normal','noisy'],
        'networkType': ['cellular'], 'isCharging': [0], 'activityState_by_motion': {'stationary':['sitting','standing']}
    },
    'at_rail_station': {
        'ps_location': ['rail_station'], 'ps_time': ['dawn','morning','forenoon','lunch','afternoon','evening','night'], 'ps_dayType': ['workday','weekend','holiday'],
        'ps_motion': ['stationary'], 'ps_phone': ['face_up','in_pocket','in_use'], 'ps_light': ['dim','normal','bright'], 'ps_sound': ['normal','noisy'],
        'networkType': ['cellular'], 'isCharging': [0], 'activityState_by_motion': {'stationary':['sitting','standing']}
    },
    'at_transit_hub': {
        'ps_location': ['transit'], 'ps_time': ['dawn','morning','forenoon','lunch','afternoon','evening','night'], 'ps_dayType': ['workday','weekend','holiday'],
        'ps_motion': ['stationary'], 'ps_phone': ['face_up','in_pocket','in_use'], 'ps_light': ['dim','normal','bright'], 'ps_sound': ['normal','noisy'],
        'networkType': ['cellular'], 'isCharging': [0], 'activityState_by_motion': {'stationary':['sitting','standing']}
    },
    'at_gym': {
        'ps_location': ['gym'], 'ps_time': ['morning','forenoon','afternoon','evening'], 'ps_dayType': ['workday','weekend','holiday'],
        'ps_motion': ['stationary','walking'], 'ps_phone': ['face_up','in_pocket','in_use'], 'ps_light': ['normal','bright'], 'ps_sound': ['normal','noisy'],
        'networkType': ['wifi','cellular'], 'isCharging': [0], 'activityState_by_motion': {'stationary':['standing','sitting'], 'walking':['standing','active']}
    },
    'at_gym_exercising': {
        'ps_location': ['gym'], 'ps_time': ['morning','forenoon','afternoon','evening'], 'ps_dayType': ['workday','weekend','holiday'],
        'ps_motion': ['running','cycling'], 'ps_phone': ['in_pocket','in_use'], 'ps_light': ['normal','bright'], 'ps_sound': ['normal','noisy'],
        'networkType': ['wifi','cellular'], 'isCharging': [0], 'activityState_by_motion': {'running':['active'], 'cycling':['active']}
    },
    'unknown_settled': {
        'ps_location': ['unknown'], 'ps_time': ['forenoon','afternoon','evening','night'], 'ps_dayType': ['workday','weekend','holiday'],
        'ps_motion': ['stationary'], 'ps_phone': ['charging','face_up','on_desk'], 'ps_light': ['dim','normal'], 'ps_sound': ['quiet','normal','noisy'],
        'networkType': ['wifi','cellular'], 'isCharging': [0,1], 'activityState_by_motion': {'stationary':['sitting','standing']}
    },
    'stationary_unknown': {
        'ps_location': ['unknown'], 'ps_time': ['forenoon','afternoon','evening','night'], 'ps_dayType': ['workday','weekend','holiday'],
        'ps_motion': ['stationary'], 'ps_phone': ['face_up','in_use','on_desk'], 'ps_light': ['dim','normal'], 'ps_sound': ['quiet','normal','noisy'],
        'networkType': ['wifi','cellular'], 'isCharging': [0], 'activityState_by_motion': {'stationary':['sitting','standing']}
    },
    'home_daytime_workday_dark': {
        'ps_location': ['home'], 'ps_time': ['forenoon','afternoon'], 'ps_dayType': ['workday'], 'ps_motion': ['stationary','walking'],
        'ps_phone': ['face_up','in_use','on_desk'], 'ps_light': ['dark'], 'ps_sound': ['quiet','normal'], 'networkType': ['wifi','cellular'],
        'isCharging': [0,1], 'activityState_by_motion': {'stationary':['sitting','standing'], 'walking':['standing','active']}
    },
    'home_evening_dark': {
        'ps_location': ['home'], 'ps_time': ['evening','night'], 'ps_dayType': ['workday','weekend'], 'ps_motion': ['stationary','walking'],
        'ps_phone': ['face_up','in_use','on_desk'], 'ps_light': ['dark'], 'ps_sound': ['quiet','normal'], 'networkType': ['wifi','cellular'],
        'isCharging': [0,1], 'activityState_by_motion': {'stationary':['sitting','standing'], 'walking':['standing','active']}
    },
    'home_sleeping': {
        'ps_location': ['home'], 'ps_time': ['sleeping'], 'ps_dayType': ['workday','weekend','holiday'], 'ps_motion': ['stationary'],
        'ps_phone': ['on_desk','face_up'], 'ps_light': ['dark','dim'], 'ps_sound': ['quiet','silent'], 'networkType': ['wifi'],
        'isCharging': [0,1], 'activityState_by_motion': {'stationary':['sleeping']}
    },
    'home_sleeping_lying': {
        'ps_location': ['home'], 'ps_time': ['sleeping'], 'ps_dayType': ['workday','weekend','holiday'], 'ps_motion': ['stationary'],
        'ps_phone': ['holding_lying'], 'ps_light': ['dark','dim'], 'ps_sound': ['quiet','silent'], 'networkType': ['wifi'],
        'isCharging': [0,1], 'activityState_by_motion': {'stationary':['sleeping','sitting']}
    },
    'home_evening': {
        'ps_location': ['home'], 'ps_time': ['evening','night'], 'ps_dayType': ['workday','weekend','holiday'], 'ps_motion': ['stationary','walking'],
        'ps_phone': ['face_up','in_use','on_desk'], 'ps_light': ['dim','normal'], 'ps_sound': ['quiet','normal'], 'networkType': ['wifi','cellular'],
        'isCharging': [0,1], 'activityState_by_motion': {'stationary':['sitting','standing'], 'walking':['standing','active']}
    },
    'home_daytime_workday_lying': {
        'ps_location': ['home'], 'ps_time': ['forenoon','afternoon'], 'ps_dayType': ['workday'], 'ps_motion': ['stationary'],
        'ps_phone': ['holding_lying'], 'ps_light': ['dim','normal'], 'ps_sound': ['quiet','normal'], 'networkType': ['wifi','cellular'],
        'isCharging': [0,1], 'activityState_by_motion': {'stationary':['sitting']}
    },
    'in_transit': {
        'ps_location': ['transit'], 'ps_time': ['morning','forenoon','afternoon','evening','night'], 'ps_dayType': ['workday','weekend','holiday'],
        'ps_motion': ['transit'], 'ps_phone': ['face_up','in_pocket','in_use'], 'ps_light': ['dim','normal','bright'], 'ps_sound': ['normal','noisy'],
        'networkType': ['cellular'], 'isCharging': [0], 'activityState_by_motion': {'transit':['sitting','standing']}
    },
    'commuting_transit_out': {
        'ps_location': ['en_route'], 'ps_time': ['dawn','morning','forenoon'], 'ps_dayType': ['workday'],
        'ps_motion': ['transit'], 'ps_phone': ['face_up','in_pocket','in_use'], 'ps_light': ['dim','normal','bright'], 'ps_sound': ['normal','noisy'],
        'networkType': ['cellular'], 'isCharging': [0], 'activityState_by_motion': {'transit':['sitting','standing']}
    },
    'commuting_walk_out': {
        'ps_location': ['en_route'], 'ps_time': ['dawn','morning'], 'ps_dayType': ['workday'],
        'ps_motion': ['walking'], 'ps_phone': ['face_up','in_pocket','in_use'], 'ps_light': ['dim','normal','bright'], 'ps_sound': ['quiet','normal'],
        'networkType': ['cellular'], 'isCharging': [0], 'activityState_by_motion': {'walking':['standing','active']}
    },
    'commuting_drive_out': {
        'ps_location': ['en_route'], 'ps_time': ['dawn','morning','forenoon'], 'ps_dayType': ['workday'],
        'ps_motion': ['driving'], 'ps_phone': ['in_pocket','face_up'], 'ps_light': ['dim','normal','bright'], 'ps_sound': ['normal','noisy'],
        'networkType': ['cellular'], 'isCharging': [0], 'activityState_by_motion': {'driving':['sitting']}
    }
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, default=Path('in_between_scenarios_spec.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--samples-per-scenario', type=int, default=1000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--scenario-ids', type=str, default='')
    p.add_argument('--smoke-samples-per-scenario', type=int, default=0)
    p.add_argument('--smoke-output', type=Path, default=None)
    p.add_argument('--smoke-report', type=Path, default=None)
    return p.parse_args()


def stable_seed(*parts: Any) -> int:
    s = '||'.join(map(str, parts))
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16)


def pick_ranked(items: list[str], rng: random.Random) -> str:
    if len(items) == 1:
        return items[0]
    if rng.random() < 0.5:
        return items[0]
    return rng.choice(items[1:])


def choose_profile(rng: random.Random) -> dict[str, Any]:
    return {
        'user_id_hash_bucket': rng.randrange(32),
        'age_bucket': rng.choice(AGE_BUCKETS),
        'sex': rng.choice(SEX_VALUES),
        'has_kids': rng.choice([0,1]),
    }


def sample_timestep(slot: str, rng: random.Random) -> tuple[int,int]:
    start, end = TIME_SLOT_RANGES[slot]
    # 5-second grid
    first = ((start + 4)//5)*5
    last = ((end - 1)//5)*5
    steps = ((last-first)//5)+1
    t = first + 5*rng.randrange(max(steps,1))
    return t//3600, t


def duration_range_for_state(state: str, scenario_id: str, motion: str) -> tuple[int,int]:
    if state == 'office_arriving':
        return 60, 1200
    if state in {'office_working','office_working_focused'}:
        return 600, 10800
    if state in {'office_overtime','office_late_overtime'}:
        return 1200, 18000
    if state in {'at_cafe','at_cafe_quiet'}:
        return 300, 5400
    if state in {'home_morning_workday','home_daytime_workday'}:
        return 300, 10800
    if state in {'at_metro','at_rail_station','at_transit_hub'}:
        return 60, 3600
    if state in {'at_gym'}:
        return 300, 1800
    if state in {'at_gym_exercising'}:
        return 600, 5400
    if state in {'unknown_settled'}:
        return 900, 7200
    if state in {'stationary_unknown'}:
        return 600, 5400
    if state in {'home_daytime_workday_dark','home_evening_dark'}:
        return 600, 10800
    if state == 'outdoor_walking':
        if scenario_id in {'POST_LUNCH_WALK_TO_OFFICE'}:
            return 120, 2400
        if scenario_id in {'AFTER_WORK_OUTDOOR_RESET'}:
            return 300, 5400
        return 120, 5400
    return 120, 7200


def activity_duration_range(state: str, motion: str, activity: str, scenario_id: str) -> tuple[int,int]:
    if activity == 'sleeping':
        return 1800, 28800
    if motion in {'running','cycling'} or activity == 'active':
        if state == 'at_gym_exercising':
            return 600, 5400
        return 120, 3600
    if motion == 'walking':
        if scenario_id in {'POST_LUNCH_WALK_TO_OFFICE','AFTER_WORK_OUTDOOR_RESET'}:
            return 300, 3600
        return 60, 2400
    if motion in {'driving','transit'}:
        return 300, 7200
    if state.startswith('office_'):
        return 300, 10800
    if state.startswith('home_'):
        return 300, 7200
    return 120, 5400


def choose_calendar(spec: dict[str, Any], rng: random.Random) -> tuple[int,int,int,str]:
    fixed = spec['fixed']
    cal_has = fixed.get('cal_hasUpcoming')
    if cal_has is None:
        mode = spec.get('calendar_mode','none')
        if mode == 'optional_light':
            cal_has = 1 if rng.random() < 0.35 else 0
        else:
            cal_has = 0
    if 'cal_event_count' in spec:
        event_count = rng.choice(spec['cal_event_count'])
    else:
        if cal_has == 0:
            event_count = rng.choice([0,0,1])
        else:
            event_count = rng.choice([1,1,2,3])
    if cal_has == 0 and event_count > 1:
        event_count = 1
    if cal_has == 0 and spec['scenario_id'] in {'HOME_DEEP_WORK_MORNING','GYM_TO_CAFE_RECOVERY','LATE_OFFICE_PRE_DEPARTURE_WRAPUP','AFTER_WORK_OUTDOOR_RESET','UNKNOWN_LONG_STAY_WITH_BOOKING'}:
        event_count = 0
    cal_in_meeting = fixed.get('cal_inMeeting', 0)
    next_loc = rng.choice(spec['next_location'])
    return cal_has, event_count, cal_in_meeting, next_loc


def choose_sms_flags(spec: dict[str, Any], rng: random.Random) -> dict[str,int]:
    out = {k:0 for k in SMS_FIELDS}
    mode = spec.get('sms_mode','none')
    if mode == 'one_of_booking':
        key = rng.choice(['sms_train_pending','sms_hotel_pending','sms_ride_pending'])
        out[key] = 1
    return out


def validate_row(spec: dict[str, Any], row: dict[str, Any]) -> list[str]:
    f = row['features']
    errs = []
    state = f['state_current']
    prof = STATE_PROFILES[state]
    if state not in spec['allowed_states']:
        errs.append('invalid state_current')
    if f['precondition'] not in spec['allowed_preconditions']:
        errs.append('invalid precondition')
    if f['ps_time'] not in spec['ps_time'] or f['ps_time'] not in prof['ps_time']:
        errs.append('invalid ps_time')
    if f['ps_dayType'] not in spec['ps_dayType'] or f['ps_dayType'] not in prof['ps_dayType']:
        errs.append('invalid ps_dayType')
    if f['ps_location'] not in prof['ps_location']:
        errs.append('invalid ps_location')
    if f['ps_motion'] not in prof['ps_motion']:
        errs.append('invalid ps_motion')
    if f['ps_phone'] not in prof['ps_phone']:
        errs.append('invalid ps_phone')
    if f['ps_light'] not in prof['ps_light']:
        errs.append('invalid ps_light')
    if f['ps_sound'] not in prof['ps_sound']:
        errs.append('invalid ps_sound')
    if f['networkType'] not in prof['networkType']:
        errs.append('invalid networkType')
    if f['isCharging'] not in prof['isCharging']:
        errs.append('invalid isCharging')
    if f['activityState'] not in prof['activityState_by_motion'][f['ps_motion']]:
        errs.append('invalid activityState for motion')
    start, end = TIME_SLOT_RANGES[f['ps_time']]
    if not (start <= f['timestep'] < end):
        errs.append('timestep not aligned to ps_time')
    if f['hour'] != f['timestep'] // 3600:
        errs.append('hour mismatch')
    if state in {'home_daytime_workday_dark','home_evening_dark'} and f['ps_light'] != 'dark':
        errs.append('dark substate without dark light')
    if state == 'at_cafe_quiet' and f['ps_sound'] != 'quiet':
        errs.append('quiet cafe without quiet sound')
    for k,v in spec['fixed'].items():
        if f[k] != v:
            errs.append(f'{k} fixed mismatch')
    if f['cal_nextLocation'] not in spec['next_location']:
        errs.append('cal_nextLocation mismatch')
    if spec.get('sms_mode') == 'one_of_booking':
        active = sum(f[k] for k in ['sms_train_pending','sms_hotel_pending','sms_ride_pending'])
        if active != 1:
            errs.append('booking sms not exactly one active')
    if f['activityDuration'] <= 0 or f['state_duration_sec'] <= 0:
        errs.append('durations must be positive')
    if f['batteryLevel'] < 20 or f['batteryLevel'] > 95:
        errs.append('battery unrealistic')
    if f['cal_eventCount'] < 0 or f['cal_eventCount'] > 3:
        errs.append('cal_eventCount unrealistic')
    return errs


def transport_from_motion(motion: str) -> str:
    return motion if motion in {'stationary','walking','running','cycling','driving','transit'} else 'unknown'


def generate_row(spec: dict[str, Any], idx: int, base_seed: int) -> dict[str, Any]:
    rng = random.Random(stable_seed(spec['scenario_id'], idx, base_seed))
    state = rng.choice(spec['allowed_states'])
    prof = STATE_PROFILES[state]
    precondition = rng.choice(spec['allowed_preconditions'])
    ps_time_choices = [x for x in spec['ps_time'] if x in prof['ps_time']]
    ps_time = rng.choice(ps_time_choices)
    hour, timestep = sample_timestep(ps_time, rng)
    ps_day = rng.choice([x for x in spec['ps_dayType'] if x in prof['ps_dayType']])
    motion = rng.choice(prof['ps_motion'])
    activity = rng.choice(prof['activityState_by_motion'][motion])
    phone = rng.choice(prof['ps_phone'])
    light = rng.choice(prof['ps_light'])
    sound = rng.choice(prof['ps_sound'])
    net = rng.choice(prof['networkType'])
    is_charging = rng.choice(prof['isCharging'])
    # make charging more coherent
    if phone == 'charging':
        is_charging = 1
    if is_charging == 1 and phone not in {'charging','on_desk','face_up'}:
        allowed_charge_phones = [x for x in prof['ps_phone'] if x in {'charging','on_desk','face_up'}]
        if allowed_charge_phones:
            phone = rng.choice(allowed_charge_phones)
    ps_location = rng.choice(prof['ps_location'])
    transport = transport_from_motion(motion)

    cal_has, cal_event_count, cal_in_meeting, next_loc = choose_calendar(spec, rng)
    wifi_lost = spec['fixed'].get('wifiLost', 0)
    wifi_cat = spec['fixed'].get('wifiLostCategory', 'unknown')
    sms_flags = choose_sms_flags(spec, rng)

    dur_lo, dur_hi = duration_range_for_state(state, spec['scenario_id'], motion)
    state_duration = rng.randint(dur_lo, dur_hi)
    act_lo, act_hi = activity_duration_range(state, motion, activity, spec['scenario_id'])
    activity_duration = rng.randint(act_lo, act_hi)
    if is_charging == 1:
        battery = rng.randint(55, 95)
    elif net == 'wifi':
        battery = rng.randint(35, 95)
    else:
        battery = rng.randint(25, 90)

    features = {
        'precondition': precondition,
        'state_current': state,
        'state_duration_sec': state_duration,
        'ps_time': ps_time,
        'hour': hour,
        'cal_hasUpcoming': cal_has,
        'ps_dayType': ps_day,
        'ps_motion': motion,
        'wifiLost': wifi_lost,
        'wifiLostCategory': wifi_cat,
        'cal_eventCount': cal_event_count,
        'cal_inMeeting': cal_in_meeting,
        'cal_nextLocation': next_loc,
        'ps_sound': sound,
        'timestep': timestep,
        'ps_location': ps_location,
        'ps_phone': phone,
        'batteryLevel': battery,
        'isCharging': is_charging,
        'networkType': net,
        'transportMode': transport,
        'activityState': activity,
        'activityDuration': activity_duration,
        'ps_light': light,
        **choose_profile(rng),
        **sms_flags,
    }

    row = {
        'scenario_id': spec['scenario_id'],
        'scenario_name': spec['scenario_name'],
        'sample_id': f"{spec['scenario_id'].lower()}_{idx:05d}",
        'features': features,
        'gt_ro': pick_ranked(spec['default_action_ids'], rng),
        'gt_app': pick_ranked(spec['default_app_categories'], rng),
    }
    errs = validate_row(spec, row)
    if errs:
        raise ValueError(f"{spec['scenario_id']} row {idx} invalid: {'; '.join(errs)}")
    return row


def main():
    args = parse_args()
    specs = json.loads(args.spec.read_text())
    if args.scenario_ids:
        wanted = {x.strip() for x in args.scenario_ids.split(',') if x.strip()}
        specs = [s for s in specs if s['scenario_id'] in wanted]
    rows = []
    meta = {'seed': args.seed, 'samples_per_scenario': args.samples_per_scenario, 'scenarios': {}, 'total_rows': 0}
    for spec in specs:
        scenario_rows = []
        for i in range(1, args.samples_per_scenario + 1):
            scenario_rows.append(generate_row(spec, i, args.seed))
        rows.extend(scenario_rows)
        meta['scenarios'][spec['scenario_id']] = {'rows_written': len(scenario_rows), 'default_action_ids': spec['default_action_ids'], 'default_app_categories': spec['default_app_categories']}
    meta['total_rows'] = len(rows)
    args.output.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + ('\n' if rows else ''), encoding='utf-8')
    args.metadata.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    if args.smoke_samples_per_scenario > 0:
        smoke_rows = []
        report_lines = ['# In-Between Scenario Smoke Test Report', '']
        total_checked = 0
        total_failed = 0
        for spec in specs:
            report_lines.append(f"## {spec['scenario_id']}")
            issues = []
            for i in range(1, args.smoke_samples_per_scenario + 1):
                row = generate_row(spec, i, args.seed)
                smoke_rows.append(row)
                errs = validate_row(spec, row)
                total_checked += 1
                if errs:
                    total_failed += 1
                    issues.append({'sample_id': row['sample_id'], 'errors': errs})
            report_lines.append(f"- checked: {args.smoke_samples_per_scenario}")
            report_lines.append(f"- issues: {len(issues)}")
            for item in issues[:5]:
                report_lines.append(f"  - {item['sample_id']}: {'; '.join(item['errors'])}")
            report_lines.append('')
        report_lines.insert(2, f"Checked rows: {total_checked}")
        report_lines.insert(3, f"Failed rows: {total_failed}")
        if args.smoke_output:
            args.smoke_output.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in smoke_rows) + ('\n' if smoke_rows else ''), encoding='utf-8')
        if args.smoke_report:
            args.smoke_report.write_text('\n'.join(report_lines), encoding='utf-8')


if __name__ == '__main__':
    main()

'''
python unique_support_to_2k_generator_state_verified_v4.py   --rules default_rules_1_english_with_preconditions_and_state_current.json   --global-action-space global_action_space_english.json   --count-report metainfo/scenario_unique_samples_count.md   --output bandit_v0_unique_support_to_2k.jsonl   --metadata metainfo/bandit_v0_unique_support_to_2k_metadata.json   --min-rows-per-scenario 2000   --seed 7
'''
#!/usr/bin/env python3
"""
Enumerate all agreed unique categorical supports per scenario, then augment scenarios
with fewer than N rows (default 2000) using realistic numeric/profile variation.

Source of categorical support:
- updated/scenario_unique_samples_count.md

Validation source:
- state_verified_generator.py helper logic (state encoder alignment + rule checks)

Behavior:
- uses updated rules with preconditions + state_current
- uses v5/global action space ranked labels
- enumerates all unique categorical combinations exactly as agreed in the count report
- if scenario support < min_rows_per_scenario, augments by varying only non-counted fields
- if scenario support >= min_rows_per_scenario, keeps all unique rows
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import state_verified_generator as base

APP_CATEGORIES = base.APP_CATEGORIES
SMS_FIELDS = base.SMS_FIELDS
TIME_SLOT_HOURS = base.TIME_SLOT_HOURS
LOCATION_CATEGORIES = base.LOCATION_CATEGORIES

# Exact slot ranges from data_spec_for_rl_english.md (start inclusive, end exclusive)
SLOT_SEC_RANGES = {
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

# Patch base validator hour buckets to the exact slot-compatible hour sets.
base.TIME_SLOT_HOURS = {
    slot: sorted({sec // 3600 for sec in range(start, end, 300)})
    for slot, (start, end) in SLOT_SEC_RANGES.items()
}
TIME_SLOT_HOURS = base.TIME_SLOT_HOURS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--global-action-space", type=Path, required=True)
    p.add_argument("--count-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--min-rows-per-scenario", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scenario-ids", type=str, default="")
    p.add_argument("--smoke-samples-per-scenario", type=int, default=0)
    p.add_argument("--smoke-output", type=Path, default=None)
    p.add_argument("--smoke-metadata", type=Path, default=None)
    p.add_argument("--smoke-report", type=Path, default=None)
    return p.parse_args()


def parse_backtick_list(s: str) -> List[str]:
    vals = re.findall(r'`([^`]+)`', s)
    if vals:
        return vals
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(',') if x.strip()]


def parse_motion_activity(cell: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for seg in cell.split(';'):
        seg = seg.strip()
        if not seg:
            continue
        motion, acts = seg.split(':', 1)
        motion = motion.strip(' `')
        for act in acts.split(','):
            act = act.strip(' `')
            if act:
                pairs.append((motion, act))
    return pairs


def parse_fixed_constraints(line: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in re.findall(r'`([^`]+)`\s*=\s*`([^`]+)`', line):
        if val in {'0', '1'}:
            out[key] = int(val)
        else:
            out[key] = val
    return out


def parse_additional_varying(line: str) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    # example: `cal_nextLocation` in (`airport`, `cafe`, ...)
    m = re.search(r'`([^`]+)`\s+in\s+\((.*)\)', line)
    if not m:
        return out
    key = m.group(1)
    vals = parse_backtick_list(m.group(2))
    out[key] = vals
    return out


def parse_count_report(path: Path) -> Dict[str, Dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: List[Dict[str, Any]] = []
    cur = None
    for line in lines:
        if line.startswith('### '):
            if cur:
                sections.append(cur)
            cur = {'header': line[4:].strip(), 'lines': []}
        elif cur is not None:
            cur['lines'].append(line)
    if cur:
        sections.append(cur)

    out: Dict[str, Dict[str, Any]] = {}
    for sec in sections:
        if not re.fullmatch(r'[A-Z0-9_]+', sec['header']):
            continue
        sid = sec['header']
        unique_line = next(l for l in sec['lines'] if 'Unique-sample count' in l)
        count = int(re.search(r'\*\*(\d[\d,]*)\*\*', unique_line).group(1).replace(',', ''))
        allowed_states = parse_backtick_list(next(l for l in sec['lines'] if 'Allowed `state_current` values' in l))
        allowed_pre = parse_backtick_list(next(l for l in sec['lines'] if 'Allowed `precondition` values' in l))
        fixed_line = next((l for l in sec['lines'] if 'Fixed relevant categorical constraints' in l), '')
        fixed = parse_fixed_constraints(fixed_line)
        add_line = next((l for l in sec['lines'] if 'Additional varying categorical rule features included' in l), '')
        additional_varying = parse_additional_varying(add_line)
        branches: List[Dict[str, Any]] = []
        for l in sec['lines']:
            if l.startswith('| `'):
                parts = [p.strip() for p in l.strip('|').split('|')]
                branches.append({
                    'state_current': parts[0].strip(' `'),
                    'precondition_values': parse_backtick_list(parts[1]),
                    'ps_time': parse_backtick_list(parts[2]),
                    'ps_dayType': parse_backtick_list(parts[3]),
                    'motion_activity_pairs': parse_motion_activity(parts[4]),
                    'ps_phone': parse_backtick_list(parts[5]),
                    'ps_light': parse_backtick_list(parts[6]),
                    'ps_sound': parse_backtick_list(parts[7]),
                    'networkType': parse_backtick_list(parts[8]),
                    'isCharging': [int(x) for x in parse_backtick_list(parts[9])],
                    'branch_count': int(parts[10]),
                })
        out[sid] = {
            'unique_count': count,
            'allowed_states': allowed_states,
            'allowed_pre': allowed_pre,
            'fixed': fixed,
            'additional_varying': additional_varying,
            'branches': branches,
        }
    return out



def _allowed_from_cond(cond) -> List[str]:
    op = cond.get('op')
    value = cond.get('value')
    if op == 'eq':
        return [str(value).strip()]
    if op == 'in':
        return [str(x).strip() for x in str(value).split(',') if str(x).strip()]
    return []


def normalize_plan_with_rules(plan: Dict[str, Any], spec: base.ScenarioSpec) -> Dict[str, Any]:
    plan = json.loads(json.dumps(plan))  # deep copy via JSON-safe content

    # categorical conditions that should directly narrow the counted support
    for cond in spec.conditions:
        key = cond.get('key')
        allowed = _allowed_from_cond(cond)
        if not allowed:
            continue
        if key == 'ps_time':
            allowed_set = set(allowed)
            for b in plan['branches']:
                filtered = [x for x in b['ps_time'] if x in allowed_set]
                b['ps_time'] = filtered if filtered else list(allowed_set)
        elif key == 'ps_dayType':
            allowed_set = set(allowed)
            for b in plan['branches']:
                filtered = [x for x in b['ps_dayType'] if x in allowed_set]
                b['ps_dayType'] = filtered if filtered else list(allowed_set)
        elif key == 'ps_motion':
            allowed_set = set(allowed)
            for b in plan['branches']:
                filtered = [pair for pair in b['motion_activity_pairs'] if pair[0] in allowed_set]
                if filtered:
                    b['motion_activity_pairs'] = filtered
                else:
                    pairs = []
                    for motion in allowed_set:
                        for act in base.motion_activity_candidates(motion, b['state_current']):
                            pairs.append((motion, act))
                    b['motion_activity_pairs'] = pairs
        elif key == 'ps_sound':
            allowed_set = set(allowed)
            for b in plan['branches']:
                filtered = [x for x in b['ps_sound'] if x in allowed_set]
                b['ps_sound'] = filtered if filtered else list(allowed_set)
        elif key == 'ps_phone':
            allowed_set = set(allowed)
            for b in plan['branches']:
                filtered = [x for x in b['ps_phone'] if x in allowed_set]
                b['ps_phone'] = filtered if filtered else list(allowed_set)
        elif key == 'ps_light':
            allowed_set = set(allowed)
            for b in plan['branches']:
                filtered = [x for x in b['ps_light'] if x in allowed_set]
                b['ps_light'] = filtered if filtered else list(allowed_set)
        elif key == 'state_current':
            allowed = set(base.normalize_state(x) for x in allowed)
            plan['branches'] = [b for b in plan['branches'] if base.normalize_state(b['state_current']) in allowed]
        elif key == 'cal_nextLocation':
            if plan.get('additional_varying') and 'cal_nextLocation' in plan['additional_varying']:
                allowed = set(allowed)
                filtered = [x for x in plan['additional_varying']['cal_nextLocation'] if x in allowed]
                plan['additional_varying']['cal_nextLocation'] = filtered if filtered else list(allowed)
            elif cond.get('op') == 'eq':
                plan['fixed']['cal_nextLocation'] = allowed[0]
        elif key in {'wifiLost','wifiLostCategory','cal_hasUpcoming','cal_inMeeting'} | set(SMS_FIELDS):
            # rule value overrides report fixed constraint
            v = allowed[0]
            if v in {'true','false'}:
                plan['fixed'][key] = 1 if v == 'true' else 0
            elif v in {'0','1'}:
                plan['fixed'][key] = int(v)
            else:
                plan['fixed'][key] = v

    # derived consistency from numeric rules
    for cond in spec.conditions:
        key, op, value = cond.get('key'), cond.get('op'), cond.get('value')
        if key == 'cal_eventCount' and str(value).isdigit():
            v = int(value)
            if op == 'eq' and v == 0:
                plan['fixed']['cal_hasUpcoming'] = 0
            elif op in {'gte','gt'} and v >= 1:
                plan['fixed']['cal_hasUpcoming'] = 1

    # drop empty branches after intersections
    new_branches = []
    for b in plan['branches']:
        if not b['precondition_values'] or not b['ps_time'] or not b['ps_dayType'] or not b['motion_activity_pairs'] or not b['ps_phone'] or not b['ps_light'] or not b['ps_sound'] or not b['networkType'] or not b['isCharging']:
            continue
        new_branches.append(b)
    plan['branches'] = new_branches
    return plan



def _slot_sec_bounds(slot: str) -> tuple[int, int]:
    return SLOT_SEC_RANGES[slot]


def _hour_bounds_from_rule(spec: base.ScenarioSpec) -> tuple[int, int]:
    lo, hi = 0, 23
    gtes = [int(c['value']) for c in spec.conditions if c.get('key') == 'hour' and c.get('op') == 'gte' and str(c.get('value')).isdigit()]
    ltes = [int(c['value']) for c in spec.conditions if c.get('key') == 'hour' and c.get('op') == 'lte' and str(c.get('value')).isdigit()]
    if gtes:
        lo = max(lo, max(gtes))
    if ltes:
        hi = min(hi, min(ltes))
    return lo, hi


def _sample_valid_timestep(slot: str, spec: base.ScenarioSpec, rng: random.Random) -> tuple[int, int]:
    start, end = _slot_sec_bounds(slot)
    lo_h, hi_h = _hour_bounds_from_rule(spec)
    valid_ranges = []
    for hour in range(lo_h, hi_h + 1):
        hstart = hour * 3600
        hend = min((hour + 1) * 3600, 24 * 3600)
        s = max(start, hstart)
        e = min(end, hend)
        if s < e:
            valid_ranges.append((s, e))
    if not valid_ranges:
        valid_ranges = [(start, end)]
    # Choose a range then a second within that range. Keep to 5-second granularity.
    s, e = rng.choice(valid_ranges)
    first = ((s + 4) // 5) * 5
    last = ((e - 1) // 5) * 5
    if last < first:
        timestep = s
    else:
        steps = ((last - first) // 5) + 1
        timestep = first + 5 * rng.randrange(steps)
    hour = timestep // 3600
    return hour, timestep


def _state_duration_range(spec: base.ScenarioSpec, state: str, motion: str) -> tuple[int, int]:
    s = base.normalize_state(state)
    # conservative realistic ranges by scenario/state family
    if s == 'office_arriving':
        lo, hi = 60, 1200
    elif s == 'office_lunch_break':
        lo, hi = 300, 3600
    elif s in {'office_working', 'office_working_focused', 'office_working_noisy'}:
        lo, hi = 600, 14400
    elif s in {'office_overtime', 'office_late_overtime', 'office_rest_day'}:
        lo, hi = 600, 21600
    elif s.startswith('commuting_') or s in {'driving', 'in_transit'}:
        lo, hi = (600, 10800) if 'LONG_DRIVE' in spec.scenario_id or s == 'driving' else (120, 5400)
    elif s in {'outdoor_walking', 'outdoor_running', 'outdoor_cycling'}:
        lo, hi = (120, 7200) if motion == 'walking' else (120, 5400)
    elif s.startswith('home_sleeping'):
        lo, hi = 1800, 28800
    elif s.startswith('home_'):
        lo, hi = 300, 14400
    elif s.startswith('at_restaurant_'):
        lo, hi = 300, 5400
    elif s in {'at_cafe', 'at_cafe_quiet'}:
        lo, hi = 300, 7200
    elif s in {'at_gym', 'at_gym_exercising'}:
        lo, hi = 300, 7200
    elif s == 'at_shopping':
        lo, hi = 300, 10800
    elif s in {'at_metro', 'at_rail_station', 'at_airport', 'at_transit_hub'}:
        lo, hi = 120, 5400
    elif s in {'stationary_unknown', 'unknown_noisy', 'unknown_dark', 'unknown_settled', 'unknown_lying'}:
        lo, hi = 300, 7200
    else:
        lo, hi = 120, 7200
    # substate persistence
    if base.normalize_state(state) in base.SUBSTATE_TRIGGER:
        lo = max(lo, 600)
    # apply numeric rule constraints if present
    for cond in spec.conditions:
        key, op, value = cond.get('key'), cond.get('op'), cond.get('value')
        if key == 'state_duration_sec' and str(value).isdigit():
            v = int(value)
            if op == 'gte': lo = max(lo, v)
            elif op == 'gt': lo = max(lo, v + 1)
            elif op == 'lte': hi = min(hi, v)
            elif op == 'lt': hi = min(hi, v - 1)
            elif op == 'eq': lo = hi = v
    hi = max(lo, hi)
    return lo, hi


def _activity_duration_range(spec: base.ScenarioSpec, state: str, activity: str, motion: str) -> tuple[int, int]:
    s = base.normalize_state(state)
    if activity == 'sleeping' or 'sleeping' in s:
        return 1800, 28800
    if motion in {'driving', 'transit'}:
        return 300, 7200
    if motion in {'running', 'cycling'} or activity == 'active':
        return 120, 5400
    if motion == 'walking':
        return 60, 3600
    if s in {'office_arriving'}:
        return 60, 1800
    if s in {'office_working', 'office_working_focused', 'office_working_noisy', 'office_overtime', 'office_late_overtime', 'office_rest_day'}:
        return 300, 14400
    return 120, 7200


def _calendar_event_count(spec: base.ScenarioSpec, fixed: dict[str, any], rng: random.Random) -> int:
    # Keep realistic small counts. If no upcoming, use 0. If upcoming, use 1-3 unless rules constrain otherwise.
    if fixed.get('cal_hasUpcoming', 0) == 0:
        base_lo, base_hi = 0, 0
    else:
        base_lo, base_hi = 1, 3
    lo, hi = base_lo, base_hi
    for cond in spec.conditions:
        key, op, value = cond.get('key'), cond.get('op'), cond.get('value')
        if key == 'cal_eventCount' and str(value).isdigit():
            v = int(value)
            if op == 'gte': lo = max(lo, v)
            elif op == 'gt': lo = max(lo, v + 1)
            elif op == 'lte': hi = min(hi, v)
            elif op == 'lt': hi = min(hi, v - 1)
            elif op == 'eq': lo = hi = v
    hi = max(lo, hi)
    return rng.randint(lo, hi)

def canonical_hour_for_slot(slot: str, spec: base.ScenarioSpec) -> int:
    lo_h, hi_h = _hour_bounds_from_rule(spec)
    allowed = [h for h in TIME_SLOT_HOURS[slot] if lo_h <= h <= hi_h] or TIME_SLOT_HOURS[slot]
    return allowed[len(allowed)//2]


def varied_hour_for_slot(slot: str, spec: base.ScenarioSpec, rng: random.Random) -> int:
    hour, _ = _sample_valid_timestep(slot, spec, rng)
    return hour


def _stable_int_seed(*parts: Any) -> int:
    s = '||'.join(str(p) for p in parts)
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16)


def _make_local_rng(seed: int, *parts: Any) -> random.Random:
    return random.Random(seed ^ _stable_int_seed(*parts))


def base_numeric_features(state: str, fixed: Dict[str, Any], spec: base.ScenarioSpec, is_charging: int, network: str, slot: str, motion: str, activity: str, mode: str, seed: int, key_parts: Tuple[Any, ...]) -> Dict[str, Any]:
    """Generate numeric/derived features with realistic variability.

    - `timestep` is sampled from the exact ps_time interval in the spec.
    - `hour` is derived from timestep, not sampled independently.
    - `state_duration_sec` is state/scenario/motion dependent and respects rule lower bounds.
    - `cal_eventCount` stays small and realistic (mostly 0-3).
    - `activityDuration` varies by activity/motion family.
    """
    local_rng = _make_local_rng(seed, *key_parts, mode) if mode == 'unique' else random.Random(seed ^ _stable_int_seed(*key_parts, random.random()))

    hour, timestep = _sample_valid_timestep(slot, spec, local_rng)

    dur_lo, dur_hi = _state_duration_range(spec, state, motion)
    state_duration = local_rng.randint(dur_lo, dur_hi)

    cal_event_count = _calendar_event_count(spec, fixed, local_rng)

    latent_next = 60 if fixed.get('cal_hasUpcoming', 0) == 1 else 240
    for cond in spec.conditions:
        key, op, value = cond.get('key'), cond.get('op'), cond.get('value')
        if key == 'cal_nextMinutes' and str(value).isdigit():
            v = int(value)
            if op == 'gte': latent_next = max(latent_next, v)
            elif op == 'gt': latent_next = max(latent_next, v + 1)
            elif op == 'lte': latent_next = min(latent_next, v)
            elif op == 'lt': latent_next = min(latent_next, v - 1)
            elif op == 'eq': latent_next = v
    low = max(0, min(latent_next, 240))
    high = max(low, min(240, latent_next + 60))
    latent_next = local_rng.randint(low, high)

    if is_charging:
        battery = local_rng.randint(55, 100)
    else:
        battery_floor = 45 if network == 'wifi' else 25
        battery = local_rng.randint(battery_floor, 95)

    activity_low, activity_high = _activity_duration_range(spec, state, activity, motion)
    activity_duration = local_rng.randint(activity_low, activity_high)

    return {
        'hour': hour,
        'timestep': timestep,
        'state_duration_sec': state_duration,
        'cal_eventCount': cal_event_count,
        '__latent_cal_nextMinutes': latent_next,
        'batteryLevel': battery,
        'activityDuration': activity_duration,
    }


def derive_location_and_transport(state: str, motion: str) -> Tuple[str, str]:
    c = base.state_constraints(state)
    # deterministic mapping for location when multiple are allowed
    # Prefer en_route for commute/driving/transit, otherwise first listed location.
    if motion in {'driving', 'transit'} and 'en_route' in c['location']:
        loc = 'en_route'
    elif state.startswith('commuting_') and 'en_route' in c['location']:
        loc = 'en_route'
    else:
        loc = c['location'][0]
    transport = motion if motion in base.TRANSPORT_MODES else 'unknown'
    if transport not in c['transport']:
        transport = c['transport'][0]
    return loc, transport



def validate_core_alignment(row: Dict[str, Any], spec: base.ScenarioSpec, branch: Dict[str, Any], fixed: Dict[str, Any], additional_varying: Dict[str, List[Any]]) -> List[str]:
    f = row['features']
    state = base.normalize_state(f['state_current'])
    c = base.state_constraints(state)
    errs: List[str] = []
    if f['ps_location'] not in c['location']:
        errs.append(f"ps_location={f['ps_location']} not in {c['location']}")
    if f['ps_motion'] not in c['motion']:
        errs.append(f"ps_motion={f['ps_motion']} not in {c['motion']}")
    if f['ps_time'] not in c['time']:
        errs.append(f"ps_time={f['ps_time']} not in {c['time']}")
    if f['ps_dayType'] not in c['day']:
        errs.append(f"ps_dayType={f['ps_dayType']} not in {c['day']}")
    # branch support checks
    if f['precondition'] not in branch['precondition_values']:
        errs.append(f"precondition={f['precondition']} not in branch preconditions")
    if f['ps_phone'] not in branch['ps_phone']:
        errs.append(f"ps_phone={f['ps_phone']} not in branch ps_phone")
    if f['ps_light'] not in branch['ps_light']:
        errs.append(f"ps_light={f['ps_light']} not in branch ps_light")
    if f['ps_sound'] not in branch['ps_sound']:
        errs.append(f"ps_sound={f['ps_sound']} not in branch ps_sound")
    if f['networkType'] not in branch['networkType']:
        errs.append(f"networkType={f['networkType']} not in branch networkType")
    if f['isCharging'] not in branch['isCharging']:
        errs.append(f"isCharging={f['isCharging']} not in branch isCharging")
    allowed_pairs = {tuple(x) for x in branch['motion_activity_pairs']}
    if (f['ps_motion'], f['activityState']) not in allowed_pairs:
        errs.append(f"(ps_motion,activityState)=({f['ps_motion']},{f['activityState']}) not in branch pairs")
    trig = c.get('substate_trigger')
    if trig:
        for k,v in trig.items():
            got = f.get(f'ps_{k}', f.get(k))
            if got != v:
                errs.append(f"substate trigger {k}={v} required but got {got}")
        if f['state_duration_sec'] < 600:
            errs.append('substate requires state_duration_sec >= 600')
    # fixed categorical constraints
    for k,v in fixed.items():
        if f.get(k) != v:
            errs.append(f"fixed constraint mismatch {k}={f.get(k)} != {v}")
    for k, vals in additional_varying.items():
        if f.get(k) not in vals:
            errs.append(f"additional varying {k}={f.get(k)} not in {vals}")
    return errs


def make_feature_row(spec: base.ScenarioSpec, fixed: Dict[str, Any], branch: Dict[str, Any], combo: Tuple[Any, ...], rng: random.Random, mode: str, row_kind: str, row_idx: int, action_type_by_id: Dict[str, str], seed: int) -> Dict[str, Any]:
    precondition, ps_time, ps_dayType, motion, activity, ps_phone, ps_light, ps_sound, networkType, isCharging, extra_dict = combo
    state = branch['state_current']
    ps_location, transport = derive_location_and_transport(state, motion)

    features: Dict[str, Any] = {
        'state_current': state,
        'precondition': precondition,
        'ps_time': ps_time,
        'ps_dayType': ps_dayType,
        'ps_motion': motion,
        'activityState': activity,
        'ps_phone': ps_phone,
        'ps_light': ps_light,
        'ps_sound': ps_sound,
        'networkType': networkType,
        'isCharging': isCharging,
        'ps_location': ps_location,
        'transportMode': transport,
        'cal_hasUpcoming': 0,
        'wifiLost': 0,
        'wifiLostCategory': 'unknown',
        'cal_inMeeting': 0,
        'cal_nextLocation': 'unknown',
    }
    # default SMS fields
    for sf in SMS_FIELDS:
        features[sf] = 0
    # fixed constraints from count report
    for k, v in fixed.items():
        features[k] = v
    for k, v in extra_dict.items():
        features[k] = v
    # numeric + profile
    key_parts = (spec.scenario_id, row_idx, state, precondition, ps_time, ps_dayType, motion, activity, ps_phone, ps_light, ps_sound, networkType, isCharging, tuple(sorted(extra_dict.items())))
    features.update(base_numeric_features(state, fixed, spec, isCharging, networkType, ps_time, motion, activity, mode=mode, seed=seed, key_parts=key_parts))
    profile = base.choose_profile(rng)
    features.update(profile)

    # choose GT with 50/50 ranked policy
    gt_ro = base.pick_ranked_label(spec.default_action_ids, rng)
    gt_app = base.pick_ranked_label(spec.default_app_categories, rng)

    row = {
        'episode_id': f"{spec.scenario_id.lower()}_{row_kind}{row_idx:06d}",
        'row_type': row_kind,
        'scenario_id': spec.scenario_id,
        'scenario_name': spec.scenario_name,
        't_in_scenario_sec': 0,
        'dt_sec': None,
        'features': features,
        'gt_ro': gt_ro,
        'gt_ro_type': action_type_by_id.get(gt_ro, 'unknown'),
        'gt_app': gt_app,
    }
    errs = validate_core_alignment(row, spec, branch, fixed, extra_dict and {k:[v] for k,v in extra_dict.items()} or {}) + base.validate_rule_conditions(row, spec)
    if errs:
        raise ValueError(f"Invalid row for {spec.scenario_id}: {'; '.join(errs)}")
    # remove latent helper before output
    row['features'].pop('__latent_cal_nextMinutes', None)
    return row


def enumerate_unique_rows(spec: base.ScenarioSpec, plan: Dict[str, Any], rng: random.Random, action_type_by_id: Dict[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 1
    for branch in plan['branches']:
        additional_items = sorted(plan.get('additional_varying', {}).items())
        additional_value_lists = [vals for _, vals in additional_items]
        combos = itertools.product(
            branch['precondition_values'],
            branch['ps_time'],
            branch['ps_dayType'],
            branch['motion_activity_pairs'],
            branch['ps_phone'],
            branch['ps_light'],
            branch['ps_sound'],
            branch['networkType'],
            branch['isCharging'],
            *additional_value_lists,
        )
        for combo_full in combos:
            precondition, ps_time, ps_dayType, motion_activity, ps_phone, ps_light, ps_sound, networkType, isCharging, *extra_vals = combo_full
            motion, activity = motion_activity
            extra_dict = {k: v for (k, _), v in zip(additional_items, extra_vals)}
            combo = (precondition, ps_time, ps_dayType, motion, activity, ps_phone, ps_light, ps_sound, networkType, isCharging, extra_dict)
            try:
                row = make_feature_row(spec, plan['fixed'], branch, combo, rng, mode='unique', row_kind='u', row_idx=idx, action_type_by_id=action_type_by_id, seed=rng.randint(0, 2**31-1))
            except ValueError:
                continue
            rows.append(row)
            idx += 1
    return rows


def augment_rows(spec: base.ScenarioSpec, plan: Dict[str, Any], unique_rows: List[Dict[str, Any]], target: int, rng: random.Random, action_type_by_id: Dict[str, str]) -> List[Dict[str, Any]]:
    rows = list(unique_rows)
    idx = 1
    # reconstruct branch lookup by state_current for convenience
    branches_by_state = {b['state_current']: b for b in plan['branches']}
    while len(rows) < target:
        base_row = unique_rows[(len(rows) - len(unique_rows)) % len(unique_rows)]
        f = base_row['features']
        branch = branches_by_state[f['state_current']]
        extra_dict = {}
        for k in plan.get('additional_varying', {}):
            extra_dict[k] = f[k]
        combo = (
            f['precondition'], f['ps_time'], f['ps_dayType'], f['ps_motion'], f['activityState'],
            f['ps_phone'], f['ps_light'], f['ps_sound'], f['networkType'], f['isCharging'], extra_dict
        )
        try:
            row = make_feature_row(spec, plan['fixed'], branch, combo, rng, mode='augment', row_kind='a', row_idx=idx, action_type_by_id=action_type_by_id, seed=rng.randint(0, 2**31-1))
        except ValueError:
            continue
        rows.append(row)
        idx += 1
    return rows


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    plans = parse_count_report(args.count_report)
    rules_by_sid = base.parse_rules(args.rules)
    scenario_defaults, action_type_by_id = base.parse_global_action_space(args.global_action_space)
    specs = base.build_scenario_specs(rules_by_sid, scenario_defaults)

    requested = {s for s in args.scenario_ids.split(',') if s} if args.scenario_ids else None
    selected_ids = [sid for sid in sorted(specs) if sid in plans and (requested is None or sid in requested)]

    all_rows: List[Dict[str, Any]] = []
    per_scenario_meta: Dict[str, Any] = {}
    mismatches = {}

    for sid in selected_ids:
        spec = specs[sid]
        plan = normalize_plan_with_rules(plans[sid], spec)
        unique_rows = enumerate_unique_rows(spec, plan, rng, action_type_by_id)
        enumerated_count = len(unique_rows)
        if enumerated_count != plan['unique_count']:
            mismatches[sid] = {'expected': plan['unique_count'], 'enumerated': enumerated_count}
        final_rows = unique_rows if enumerated_count >= args.min_rows_per_scenario else augment_rows(spec, plan, unique_rows, args.min_rows_per_scenario, rng, action_type_by_id)
        all_rows.extend(final_rows)
        per_scenario_meta[sid] = {
            'unique_count_reported': plan['unique_count'],
            'unique_count_enumerated': enumerated_count,
            'final_rows_written': len(final_rows),
            'augmented': len(final_rows) > enumerated_count,
        }

    args.output.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in all_rows) + ('\n' if all_rows else ''), encoding='utf-8')

    metadata = {
        'rules_source': str(args.rules),
        'global_action_space_source': str(args.global_action_space),
        'count_report_source': str(args.count_report),
        'min_rows_per_scenario': args.min_rows_per_scenario,
        'seed': args.seed,
        'included_scenarios': selected_ids,
        'total_rows_written': len(all_rows),
        'count_mismatches': mismatches,
        'per_scenario': per_scenario_meta,
        'gt_policy': {
            'ro_default_prob': 0.5,
            'ro_remaining_policy': 'uniform over remaining ranked actions',
            'app_default_prob': 0.5,
            'app_remaining_policy': 'uniform over remaining ranked app categories',
        },
        'generation_policy': {
            'unique_support_source': 'scenario_unique_samples_count.md',
            'enumerate_all_unique_categorical_supports': True,
            'augment_if_less_than_min_rows': True,
            'keep_all_if_greater_than_or_equal_to_min_rows': True,
        }
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    if args.smoke_samples_per_scenario and args.smoke_samples_per_scenario > 0:
        smoke_rows: List[Dict[str, Any]] = []
        validation = {'checked_rows': 0, 'failed_rows': 0, 'scenarios': {}}
        by_sid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in all_rows:
            by_sid[r['scenario_id']].append(r)
        for sid in selected_ids:
            sample_rows = by_sid[sid][:args.smoke_samples_per_scenario]
            issues = []
            for r in sample_rows:
                rr = json.loads(json.dumps(r))
                # reconstruct a latent cal_nextMinutes that satisfies the rule constraints when needed
                latent = 60 if rr['features'].get('cal_hasUpcoming', 0) else 240
                for cond in specs[sid].conditions:
                    if cond.get('key') == 'cal_nextMinutes' and str(cond.get('value')).isdigit():
                        v = int(cond['value'])
                        op = cond.get('op')
                        if op == 'lte':
                            latent = min(latent, v)
                        elif op == 'lt':
                            latent = min(latent, v - 1)
                        elif op == 'gte':
                            latent = max(latent, v)
                        elif op == 'gt':
                            latent = max(latent, v + 1)
                rr['features']['__latent_cal_nextMinutes'] = latent
                # recover branch for custom validation
                plan_branch = next(b for b in plans[sid]['branches'] if b['state_current']==rr['features']['state_current'])
                add_var = {k: plans[sid]['additional_varying'][k] for k in plans[sid].get('additional_varying', {})}
                errs = validate_core_alignment(rr, specs[sid], plan_branch, plans[sid]['fixed'], add_var) + base.validate_rule_conditions(rr, specs[sid])
                rr['features'].pop('__latent_cal_nextMinutes', None)
                validation['checked_rows'] += 1
                if errs:
                    validation['failed_rows'] += 1
                    issues.append({'episode_id': r['episode_id'], 'errors': errs})
                smoke_rows.append(r)
            validation['scenarios'][sid] = {'checked': len(sample_rows), 'issues': issues}
        if args.smoke_output:
            args.smoke_output.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in smoke_rows) + ('\n' if smoke_rows else ''), encoding='utf-8')
        if args.smoke_metadata:
            args.smoke_metadata.write_text(json.dumps(validation, indent=2), encoding='utf-8')
        if args.smoke_report:
            lines = ['# Unique Support Smoke Test Validation Report', '', f"Checked rows: {validation['checked_rows']}", f"Failed rows: {validation['failed_rows']}", '']
            for sid in selected_ids:
                issues = validation['scenarios'][sid]['issues']
                lines.append(f"## {sid}")
                lines.append(f"- checked: {validation['scenarios'][sid]['checked']}")
                lines.append(f"- issues: {len(issues)}")
                if issues:
                    for item in issues[:5]:
                        lines.append(f"  - {item['episode_id']}: {'; '.join(item['errors'])}")
                lines.append('')
            args.smoke_report.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

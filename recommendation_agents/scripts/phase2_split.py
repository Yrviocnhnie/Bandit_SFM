#!/usr/bin/env python3
"""Phase-2 data splitter.

Selects test samples from original data matching phase-2 context variations.
- 6 contexts have exact matches → used as-is
- 3 contexts have no exact match (calendar fields don't exist) →
  pick nearest sample, patch only the impossible fields, verify all
  feature consistency rules (state-code encoding, hour/ps_time alignment, etc.)
"""
import argparse, json, copy, random, math
from collections import defaultdict
from pathlib import Path

DEFAULT_DATA = "/data00/liyao/projects/sfm/Bandit_SFM/data/bandit_v0_65_scenaroios_unique_support_to_2k_samples_office_working_alias.jsonl"
DEFAULT_OUT_DIR = "/home/mohan/Bandit_SFM/recommendation_agents/artifacts/phase2"
DEFAULT_SAMPLES_PER_CONTEXT = 3
DEFAULT_SEED = 42

# Default single user profile (most common profile in the original data)
DEFAULT_PROFILE = {
    "user_id_hash_bucket": "b11",
    "age_bucket": "25_34",
    "sex": "male",
    "has_kids": 0,
}
PROFILE_FIELDS = ["user_id_hash_bucket", "age_bucket", "sex", "has_kids"]

# ── Time slot → hour ranges (from data_spec_for_rl_english.md §1.1) ──
TIME_SLOT_HOURS = {
    "sleeping":   (0, 5),      # 0:00–5:30
    "dawn":       (5, 6),      # 5:30–7:00 → hours 5,6
    "morning":    (7, 8),      # 7:00–9:00 → hours 7,8
    "forenoon":   (9, 11),     # 9:00–11:30 → hours 9,10,11
    "lunch":      (11, 13),    # 11:30–13:30 → hours 11,12,13
    "afternoon":  (13, 16),    # 13:30–17:00 → hours 13,14,15,16
    "evening":    (17, 19),    # 17:00–19:30 → hours 17,18,19
    "night":      (19, 21),    # 19:30–22:00 → hours 19,20,21
    "late_night": (22, 23),    # 22:00–0:00 → hours 22,23
}

# ── State-code encoding rules (from data_spec §2) ──
# state_current → required (location, motion, time_slots, day_types)
STATE_CODE_RULES = {
    "office_arriving":   {"ps_location": "work", "ps_time": ["dawn", "morning"], "ps_dayType": ["workday"]},
    "office_working":    {"ps_location": "work", "ps_time": ["forenoon", "afternoon"], "ps_dayType": ["workday"]},
    "office_lunch_break":{"ps_location": "work", "ps_time": ["lunch"], "ps_dayType": ["workday"]},
    "office_overtime":   {"ps_location": "work", "ps_time": ["evening", "night"], "ps_dayType": ["workday"]},
    "at_cafe_quiet":     {"ps_location": "cafe", "ps_sound": "quiet"},  # substate of at_cafe
}


def get_feat(row, key):
    return row.get("features", row).get(key)


def set_feat(row, key, value):
    if "features" in row:
        row["features"][key] = value
    else:
        row[key] = value


def verify_sample(row, context_id):
    """Verify all feature consistency rules on a sample. Returns list of issues."""
    issues = []
    feat = row.get("features", row)

    state = feat.get("state_current", "")
    ps_time = feat.get("ps_time", "")
    ps_location = feat.get("ps_location", "")
    ps_dayType = feat.get("ps_dayType", "")
    ps_motion = feat.get("ps_motion", "")
    ps_sound = feat.get("ps_sound", "")
    hour = feat.get("hour")
    timestep = feat.get("timestep")

    # 1. hour must fall within ps_time range
    if ps_time in TIME_SLOT_HOURS and hour is not None:
        lo, hi = TIME_SLOT_HOURS[ps_time]
        if not (lo <= hour <= hi):
            issues.append(f"hour={hour} outside ps_time={ps_time} range [{lo}-{hi}]")

    # 2. timestep must be consistent with hour (timestep = seconds since midnight)
    if hour is not None and timestep is not None:
        expected_hour = timestep // 3600
        if expected_hour != hour:
            issues.append(f"timestep={timestep} → hour {expected_hour}, but hour={hour}")

    # 3. state-code encoding rules
    if state in STATE_CODE_RULES:
        rules = STATE_CODE_RULES[state]
        if "ps_location" in rules and ps_location != rules["ps_location"]:
            issues.append(f"state={state} requires ps_location={rules['ps_location']}, got {ps_location}")
        if "ps_time" in rules and ps_time not in rules["ps_time"]:
            issues.append(f"state={state} requires ps_time in {rules['ps_time']}, got {ps_time}")
        if "ps_dayType" in rules and ps_dayType not in rules["ps_dayType"]:
            issues.append(f"state={state} requires ps_dayType in {rules['ps_dayType']}, got {ps_dayType}")
        if "ps_sound" in rules and ps_sound != rules["ps_sound"]:
            issues.append(f"state={state} requires ps_sound={rules['ps_sound']}, got {ps_sound}")

    # 4. cal_hasUpcoming consistency with cal_eventCount
    cal_has = feat.get("cal_hasUpcoming")
    cal_count = feat.get("cal_eventCount")
    if cal_has == 0 and cal_count is not None and cal_count > 0:
        issues.append(f"cal_hasUpcoming=0 but cal_eventCount={cal_count}")
    if cal_has == 1 and cal_count is not None and cal_count == 0:
        issues.append(f"cal_hasUpcoming=1 but cal_eventCount=0")

    # 5. cal_nextLocation should be set if cal_hasUpcoming=1
    cal_next = feat.get("cal_nextLocation", "unknown")
    if cal_has == 1 and cal_next == "unknown":
        # Not strictly wrong, but worth noting
        pass

    return issues


def patch_and_verify(row, patches, context_id):
    """Apply patches to a deep copy and run verification. Returns (patched_row, issues)."""
    patched = copy.deepcopy(row)
    feat = patched.get("features", patched)

    for k, v in patches.items():
        feat[k] = v

    # For ps_time patches, also fix hour and timestep to be consistent
    if "ps_time" in patches:
        new_ps_time = patches["ps_time"]
        if new_ps_time in TIME_SLOT_HOURS:
            lo, hi = TIME_SLOT_HOURS[new_ps_time]
            current_hour = feat.get("hour")
            if current_hour is None or not (lo <= current_hour <= hi):
                # Pick middle hour of range
                new_hour = (lo + hi) // 2
                feat["hour"] = new_hour
                # Recalculate timestep: keep the minutes/seconds from original
                old_ts = feat.get("timestep", 0)
                old_minutes_seconds = old_ts % 3600
                feat["timestep"] = new_hour * 3600 + old_minutes_seconds

    issues = verify_sample(patched, context_id)
    return patched, issues


# ── Context definitions ─────────────────────────────────────────────
# exact_match: all features that must match from original data
# patches: only for impossible contexts — minimal fields to overwrite
CONTEXTS = [
    # ── 6 contexts with exact matches ──
    {"context_id": "ARRIVE_OFFICE_A", "label": "Meeting-heavy arrival",
     "scenario_id": "ARRIVE_OFFICE", "gt_ro": "O_SHOW_SCHEDULE",
     "exact_match": {"state_current": "office_arriving", "precondition": "commuting_transit_out",
                     "ps_time": "morning", "ps_dayType": "workday", "ps_motion": "walking",
                     "ps_location": "work", "cal_hasUpcoming": 1, "cal_eventCount": 3, "cal_inMeeting": 0},
     "patches": {}},

    {"context_id": "LEAVE_OFFICE_A", "label": "Driving commute",
     "scenario_id": "LEAVE_OFFICE", "gt_ro": "O_SHOW_COMMUTE_TRAFFIC",
     "exact_match": {"state_current": "commuting_drive_home", "precondition": "office_overtime",
                     "ps_time": "night", "ps_dayType": "workday", "ps_motion": "driving",
                     "ps_location": "en_route", "wifiLost": 1, "wifiLostCategory": "work",
                     "networkType": "cellular"},
     "patches": {}},

    {"context_id": "LEAVE_OFFICE_B", "label": "Walking commute",
     "scenario_id": "LEAVE_OFFICE", "gt_ro": "O_SHOW_WEATHER",
     "exact_match": {"state_current": "outdoor_walking", "precondition": "office_working",
                     "ps_time": "evening", "ps_dayType": "workday", "ps_motion": "walking",
                     "ps_location": "outdoor", "wifiLost": 1, "wifiLostCategory": "work"},
     "patches": {}},

    {"context_id": "OFFICE_LUNCH_OUT_A", "label": "Ready to pay",
     "scenario_id": "OFFICE_LUNCH_OUT", "gt_ro": "O_SHOW_PAYMENT_QR",
     "exact_match": {"state_current": "at_restaurant_lunch", "precondition": "office_working",
                     "ps_time": "lunch", "ps_dayType": "workday", "ps_motion": "stationary",
                     "ps_location": "restaurant", "wifiLost": 1, "wifiLostCategory": "work",
                     "activityState": "sitting"},
     "patches": {}},

    {"context_id": "OFFICE_LUNCH_OUT_B", "label": "Stepping out / break",
     "scenario_id": "OFFICE_LUNCH_OUT", "gt_ro": "R_ENJOY_LEISURE_MOMENT",
     "exact_match": {"state_current": "outdoor_walking", "precondition": "office_lunch_break",
                     "ps_time": "lunch", "ps_dayType": "workday", "ps_motion": "walking",
                     "ps_location": "outdoor", "wifiLost": 1, "wifiLostCategory": "work",
                     "activityState": "active"},
     "patches": {}},

    {"context_id": "CAFE_QUIET_B", "label": "Deep focus",
     "scenario_id": "CAFE_QUIET", "gt_ro": "O_TURN_ON_SILENT_MODE",
     "exact_match": {"state_current": "at_cafe_quiet", "precondition": "at_cafe",
                     "ps_time": "afternoon", "ps_dayType": "workday", "ps_motion": "stationary",
                     "ps_location": "cafe", "ps_sound": "quiet", "ps_phone": "face_up",
                     "cal_hasUpcoming": 0, "cal_eventCount": 0},
     "patches": {}},

    # ── 3 contexts that need minimal patching ──

    # ARRIVE_OFFICE_B: all ARRIVE_OFFICE have cal_hasUpcoming=1
    # Nearest match: office_working + commuting_walk_out + forenoon + workday + stationary + on_desk + sitting
    # Patch: cal_hasUpcoming → 0, cal_eventCount → 0
    {"context_id": "ARRIVE_OFFICE_B", "label": "Task-focused start",
     "scenario_id": "ARRIVE_OFFICE", "gt_ro": "O_SHOW_TODAY_TODO",
     "exact_match": {"state_current": "office_working", "precondition": "commuting_walk_out",
                     "ps_time": "forenoon", "ps_dayType": "workday", "ps_motion": "stationary",
                     "ps_location": "work", "ps_phone": "on_desk", "activityState": "sitting"},
     "patches": {"cal_hasUpcoming": 0, "cal_eventCount": 0}},

    # ARRIVE_OFFICE_C: same calendar issue
    # Nearest match: office_arriving + commuting_walk_out + dawn + workday + walking + in_pocket
    # Patch: cal_hasUpcoming → 0
    # Note: cal_eventCount not specified in phase-2 doc for this context, but if cal_hasUpcoming=0
    #       then cal_eventCount should also be 0 for consistency
    {"context_id": "ARRIVE_OFFICE_C", "label": "Soft start morning",
     "scenario_id": "ARRIVE_OFFICE", "gt_ro": "R_PLAN_DAY_OVER_COFFEE",
     "exact_match": {"state_current": "office_arriving", "precondition": "commuting_walk_out",
                     "ps_time": "dawn", "ps_dayType": "workday", "ps_motion": "walking",
                     "ps_phone": "in_pocket"},
     "patches": {"cal_hasUpcoming": 0, "cal_eventCount": 0}},

    # CAFE_QUIET_A: no forenoon, no cal_hasUpcoming=1, no cal_eventCount=1, no cal_nextLocation=work
    # Nearest match: at_cafe_quiet + at_cafe + workday + stationary + cafe + quiet
    # Patch: ps_time → forenoon (+ fix hour/timestep), cal_hasUpcoming → 1,
    #        cal_eventCount → 1, cal_nextLocation → work
    {"context_id": "CAFE_QUIET_A", "label": "Upcoming meeting",
     "scenario_id": "CAFE_QUIET", "gt_ro": "O_SHOW_SCHEDULE",
     "exact_match": {"state_current": "at_cafe_quiet", "precondition": "at_cafe",
                     "ps_dayType": "workday", "ps_motion": "stationary",
                     "ps_location": "cafe", "ps_sound": "quiet"},
     "patches": {"ps_time": "forenoon", "cal_hasUpcoming": 1,
                 "cal_eventCount": 1, "cal_nextLocation": "work"}},
]


def main():
    parser = argparse.ArgumentParser(description="Phase-2 train/test data splitter")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Path to original raw JSONL")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR,
                        help="Directory to write train/test/report files")
    parser.add_argument("--samples-per-context", type=int, default=DEFAULT_SAMPLES_PER_CONTEXT,
                        help="How many test samples to draw per context (default: 3)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for sample selection (default: 42)")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any context has fewer candidates than requested")
    # Profile normalization: overwrite all train+test rows with a single profile
    parser.add_argument("--profile-user-id", default=DEFAULT_PROFILE["user_id_hash_bucket"],
                        help=f"user_id_hash_bucket value (default: {DEFAULT_PROFILE['user_id_hash_bucket']})")
    parser.add_argument("--profile-age", default=DEFAULT_PROFILE["age_bucket"],
                        help=f"age_bucket value (default: {DEFAULT_PROFILE['age_bucket']})")
    parser.add_argument("--profile-sex", default=DEFAULT_PROFILE["sex"],
                        help=f"sex value (default: {DEFAULT_PROFILE['sex']})")
    parser.add_argument("--profile-has-kids", type=int, default=DEFAULT_PROFILE["has_kids"],
                        help=f"has_kids value 0/1 (default: {DEFAULT_PROFILE['has_kids']})")
    parser.add_argument("--no-normalize-profile", action="store_true",
                        help="Disable profile normalization (keep original profile fields)")
    args = parser.parse_args()

    profile = {
        "user_id_hash_bucket": args.profile_user_id,
        "age_bucket": args.profile_age,
        "sex": args.profile_sex,
        "has_kids": args.profile_has_kids,
    }
    normalize_profile = not args.no_normalize_profile

    samples_per_context = args.samples_per_context
    if samples_per_context <= 0:
        parser.error("--samples-per-context must be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Load all rows
    all_rows = []
    rows_by_scenario = defaultdict(list)
    with open(args.data) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            row["_orig_idx"] = i
            all_rows.append(row)
            rows_by_scenario[row["scenario_id"]].append(row)

    print(f"Loaded {len(all_rows)} rows across {len(rows_by_scenario)} scenarios")
    print(f"Samples per context: {samples_per_context} | Seed: {args.seed}")
    if normalize_profile:
        print(f"Profile normalization: ON → {profile}\n")
    else:
        print(f"Profile normalization: OFF (original profiles preserved)\n")

    test_indices = set()
    test_rows = []
    selection_report = []
    insufficient = []

    for ctx in CONTEXTS:
        sid = ctx["scenario_id"]
        exact_match = ctx["exact_match"]
        patches = ctx.get("patches", {})

        # Find candidates matching exact_match fields
        candidates = [r for r in rows_by_scenario[sid]
                      if all(get_feat(r, k) == v for k, v in exact_match.items())]

        if len(candidates) < samples_per_context:
            msg = f"{ctx['context_id']} — only {len(candidates)} candidates (requested {samples_per_context})"
            print(f"WARNING: {msg}")
            insufficient.append(msg)

        rng.shuffle(candidates)
        selected = candidates[:samples_per_context]

        needs_patch = bool(patches)
        ctx_report = {
            "context_id": ctx["context_id"],
            "label": ctx["label"],
            "scenario_id": sid,
            "gt_ro": ctx["gt_ro"],
            "candidates_found": len(candidates),
            "selected_count": len(selected),
            "needs_patch": needs_patch,
            "patched_fields": list(patches.keys()) if needs_patch else [],
            "selected_episodes": [],
            "verification": [],
        }

        for row in selected:
            idx = row["_orig_idx"]
            ep = row.get("episode_id", "?")

            if needs_patch:
                final, issues = patch_and_verify(row, patches, ctx["context_id"])
            else:
                final = copy.deepcopy(row)
                issues = verify_sample(final, ctx["context_id"])

            final.pop("_orig_idx", None)

            # Apply profile normalization
            if normalize_profile:
                feat = final.get("features", final)
                for pk, pv in profile.items():
                    feat[pk] = pv

            final["phase2_context_id"] = ctx["context_id"]
            final["phase2_gt_ro"] = ctx["gt_ro"]
            final["phase2_label"] = ctx["label"]
            if needs_patch:
                final["phase2_patched_fields"] = list(patches.keys())

            test_indices.add(idx)
            test_rows.append(final)

            sample_info = {"episode_id": ep, "issues": issues}
            ctx_report["selected_episodes"].append(ep)
            ctx_report["verification"].append(sample_info)

            if issues:
                print(f"  ISSUE in {ctx['context_id']} / {ep}: {issues}")

        selection_report.append(ctx_report)

    if args.strict and insufficient:
        print(f"\nERROR: --strict mode and insufficient candidates:")
        for m in insufficient:
            print(f"  {m}")
        raise SystemExit(1)

    # Write test file
    test_path = out_dir / "test_phase2.raw.jsonl"
    with test_path.open("w") as f:
        for row in test_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    # Write train file (apply profile normalization on the fly if enabled)
    train_path = out_dir / "train_phase2.raw.jsonl"
    train_count = 0
    with open(args.data) as fin, train_path.open("w") as fout:
        for i, line in enumerate(fin):
            if i in test_indices:
                continue
            if normalize_profile:
                row = json.loads(line)
                feat = row.get("features", row)
                for pk, pv in profile.items():
                    feat[pk] = pv
                fout.write(json.dumps(row, sort_keys=True) + "\n")
            else:
                fout.write(line)
            train_count += 1

    # Write report
    report = {
        "input_data": args.data,
        "total_rows": len(all_rows),
        "test_rows": len(test_rows),
        "train_rows": train_count,
        "samples_per_context": samples_per_context,
        "seed": args.seed,
        "profile_normalized": normalize_profile,
        "profile": profile if normalize_profile else None,
        "contexts": selection_report,
    }
    report_path = out_dir / "test_phase2_selection_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Total: {len(all_rows)} | Test: {len(test_rows)} | Train: {train_count}")
    print(f"{'='*70}")
    for cr in selection_report:
        patched = f" [PATCHED: {cr['patched_fields']}]" if cr["needs_patch"] else " [exact]"
        marker = " ⚠" if cr["selected_count"] < samples_per_context else ""
        print(f"  {cr['context_id']:25s} | cands={cr['candidates_found']:5d} | "
              f"sel={cr['selected_count']}{marker}{patched}")

    all_issues = [(cr["context_id"], si["episode_id"], si["issues"])
                  for cr in selection_report for si in cr["verification"] if si["issues"]]
    if all_issues:
        print(f"\n*** VERIFICATION ISSUES ({len(all_issues)}) ***")
        for cid, ep, iss in all_issues:
            print(f"  {cid} / {ep}: {iss}")
    else:
        print(f"\n✓ All {len(test_rows)} samples passed verification.")

    print(f"\nFiles:")
    print(f"  {test_path}")
    print(f"  {train_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()

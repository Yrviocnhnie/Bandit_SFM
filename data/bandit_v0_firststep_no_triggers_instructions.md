# How to generate the new v0 dataset

This guide is for the **latest agreed generation mode**:

- ignore all scenarios that have **non-`none` recommendation triggers**
- generate **only the first prediction timestep** of each episode
- keep the **v0** feature set only
- keep `precondition`
- remove `transportMode`
- remove `cal_nextMinutes`
- use **70% default / 30% alternative** GT targets

## Current scenario filtering

When the script parses the current rules + action files, it excludes **4 trigger-based scenarios out of 65**:

- `OFFICE_AFTERNOON`
- `WEEKEND_OVERTIME`
- `HOME_EVENING`
- `GYM_WORKOUT`

So the generator writes data for **61 scenarios**.

## What the script outputs

The script writes:

1. **JSONL data file**
   - one JSON object per episode
   - one row only, at the **first prediction timestep**
2. **metadata JSON**
   - generation settings
   - ignored scenarios
   - included scenarios
   - per-scenario default/alternative counts
3. **optional plan JSON**
   - state options
   - precondition options
   - time-window heuristic for each included scenario

## Main script

Use this file:

- `generate_bandit_v0_firststep_no_triggers.py`

## Recommended command for the full dataset

This generates **about 1,000 episodes for each included scenario** into one JSONL file:

```bash
python generate_bandit_v0_firststep_no_triggers.py \
  --rules default_rules_1_english.json \
  --actions scenario_recommendation_actions_v4_english.md \
  --output bandit_v0_firststep_no_triggers_1000eps.jsonl \
  --metadata bandit_v0_firststep_no_triggers_1000eps_metadata.json \
  --plan-output scenario_generation_plan_v0_firststep_no_triggers.json \
  --episodes-per-scenario 1000 \
  --default-ratio 0.70 \
  --seed 7
```

## What “1000 episodes per scenario” means here

Because we now generate **only the first timestep**, the row count is easy:

- **1 episode = 1 row**
- **1000 episodes per scenario = 1000 rows per scenario**

Since the script currently includes **61 scenarios**, the full dataset will contain:

- **61,000 rows total**

## Output label policy

Each episode row always contains a prediction target, because we now keep only the first prediction timestep.

### RO target

- **70%** of episodes use the scenario's **default RO recommendation**
- **30%** of episodes use one of the **other RO recommendations in the same scenario**

### App target

- **70%** of episodes use the scenario's **default app category**
- **30%** of episodes use an **alternative app category from the other categories**

The script uses an **exact shuffled allocation per scenario**, so the split is much closer to the requested 70/30 than purely random sampling.

## Features included in each row

The script generates the approved **v0** features:

- `scenarioId`
- `precondition`
- `state_current`
- `state_duration_sec`
- `ps_time`
- `hour`
- `cal_hasUpcoming`
- `ps_dayType`
- `ps_motion`
- `wifiLost`
- `wifiLostCategory`
- `cal_eventCount`
- `cal_inMeeting`
- `cal_nextLocation`
- `ps_sound`
- `timestep`
- `ps_location`
- `ps_phone`
- `batteryLevel`
- `isCharging`
- `networkType`
- `activityState`
- `activityDuration`
- `user_id_hash_bucket`
- `age_bucket`
- `sex`
- `has_kids`
- `sms_delivery_pending`
- `sms_train_pending`
- `sms_flight_pending`
- `sms_hotel_pending`
- `sms_movie_pending`
- `sms_hospital_pending`
- `sms_ride_pending`

The script intentionally does **not** include:

- history features
- `transportMode`
- `cal_nextMinutes`

## Generate a small subset first

Example: generate only `ARRIVE_OFFICE` and `OFFICE_LUNCH_OUT` with 10 episodes each.

```bash
python generate_bandit_v0_firststep_no_triggers.py \
  --rules default_rules_1_english.json \
  --actions scenario_recommendation_actions_v4_english.md \
  --output subset_verify.jsonl \
  --metadata subset_verify_metadata.json \
  --plan-output subset_verify_plan.json \
  --episodes-per-scenario 10 \
  --default-ratio 0.70 \
  --seed 7 \
  --scenario-ids ARRIVE_OFFICE,OFFICE_LUNCH_OUT
```

## Output schema example

```json
{
  "episode_id": "arrive_office_ep0001",
  "scenario_id": "ARRIVE_OFFICE",
  "scenario_name": "Arrive at Office",
  "t_in_scenario_sec": 0,
  "dt_sec": null,
  "features": {
    "scenarioId": "ARRIVE_OFFICE",
    "precondition": "none",
    "state_current": "office_arriving",
    "state_duration_sec": 0,
    "ps_time": "morning",
    "hour": 8,
    "cal_hasUpcoming": 1,
    "ps_dayType": "workday",
    "ps_motion": "stationary",
    "wifiLost": 0,
    "wifiLostCategory": "unknown",
    "cal_eventCount": 3,
    "cal_inMeeting": 0,
    "cal_nextLocation": "work",
    "ps_sound": "normal",
    "timestep": 32100,
    "ps_location": "work",
    "ps_phone": "in_use",
    "batteryLevel": 83,
    "isCharging": 0,
    "networkType": "wifi",
    "activityState": "standing",
    "activityDuration": 0,
    "user_id_hash_bucket": "b07",
    "age_bucket": "25_34",
    "sex": "female",
    "has_kids": 0,
    "sms_delivery_pending": 0,
    "sms_train_pending": 0,
    "sms_flight_pending": 0,
    "sms_hotel_pending": 0,
    "sms_movie_pending": 0,
    "sms_hospital_pending": 0,
    "sms_ride_pending": 0
  },
  "gt_ro": "arrive_office_schedule",
  "gt_app": "productivity"
}
```

## Practical note

Because this mode emits one row per episode, the dataset is ideal for your current **v0 contextual-bandit training** objective:

- each row is a single scenario-conditioned decision point
- every row has a non-`NONE` target
- row counts are easy to control exactly per scenario

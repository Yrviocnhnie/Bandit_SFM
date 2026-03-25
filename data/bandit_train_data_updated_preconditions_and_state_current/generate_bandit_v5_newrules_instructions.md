# Updated v5 data-generation script

## Files
- Rules with added preconditions and added `state_current` for previously missing scenarios:
  - `default_rules_1_english_with_preconditions_and_state_current.json`
- Generator:
  - `generate_bandit_v0_firststep_no_triggers_v5_newrules.py`

## What changed
- Uses the updated rules file where the 13 scenarios that previously had no `state_current` now have realistic `state_current` candidate lists.
- Uses `global_action_space_english.json` for GT label selection.
- Keeps the first-timestep-only generation mode.
- Keeps the current ignore list for trigger-based scenarios.
- Uses a **50/50 ranked-label policy**:
  - `gt_ro`: first action in `defaultActionIds` has 50% probability; the remaining 50% is distributed uniformly across the rest.
  - `gt_app`: first category in `defaultAppCategories` has 50% probability; the remaining 50% is distributed uniformly across the rest.

## Example command
```bash
python generate_bandit_v0_firststep_no_triggers_v5_newrules.py   --rules default_rules_1_english_with_preconditions_and_state_current.json   --actions-md scenario_recommendation_actions_v5_english.md   --global-action-space global_action_space_english.json   --output bandit_v0_v5_1000eps.jsonl   --metadata bandit_v0_v5_1000eps_metadata.json   --plan-output scenario_generation_plan_v5.json   --episodes-per-scenario 1000   --seed 7
```

## Output
- One JSON object per episode in JSONL.
- The current approved mode still ignores these 4 scenarios:
  - `OFFICE_AFTERNOON`
  - `WEEKEND_OVERTIME`
  - `HOME_EVENING`
  - `GYM_WORKOUT`

So the default full run generates:
- **61 included scenarios**
- **1000 episodes per scenario**
- **61,000 rows**

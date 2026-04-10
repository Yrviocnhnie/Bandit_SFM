# Feedback Propagation Experiment

- artifact_dir: `artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split`
- device: `cpu`
- n_values: `2000`
- similarity_thresholds: `0.800`

## Locked Feedback Items

| scenario | feedback | target_position | target_action_id | anchor_episode_id | baseline_top3 |
| --- | --- | ---: | --- | --- | --- |
| `ARRIVE_OFFICE` | `like` | 1 | `O_SHOW_SCHEDULE` | `arrive_office_u000002` | `O_SHOW_SCHEDULE , O_SHOW_TODAY_TODO , R_PLAN_DAY_OVER_COFFEE` |
| `OFFICE_LUNCH_OUT` | `dislike` | 1 | `O_SHOW_NEARBY_OPTIONS` | `office_lunch_out_a000002` | `O_SHOW_NEARBY_OPTIONS , R_MEAL_BREAK , O_SHOW_PAYMENT_QR` |
| `OFFICE_WORKING` | `like` | 2 | `O_SHOW_SCHEDULE` | `office_afternoon_u000008` | `O_SHOW_TODAY_TODO , O_SHOW_SCHEDULE , R_WORK_BREAK_AND_STRETCH` |
| `LEAVE_OFFICE` | `dislike` | 2 | `R_CLOCK_OUT_BEFORE_LEAVING` | `leave_office_a000009` | `O_SHOW_WEATHER , R_CLOCK_OUT_BEFORE_LEAVING , O_SHOW_COMMUTE_TRAFFIC` |
| `CAFE_STAY` | `like` | 3 | `O_SHOW_PAYMENT_QR` | `cafe_stay_u000004` | `O_SHOW_SCHEDULE , R_ENJOY_LEISURE_MOMENT , O_SHOW_PAYMENT_QR` |
| `OFFICE_TO_CAFE` | `dislike` | 3 | `O_SHOW_NEARBY_OPTIONS` | `office_to_cafe_u000003` | `R_ENJOY_LEISURE_MOMENT , O_SHOW_PAYMENT_QR , O_SHOW_NEARBY_OPTIONS` |

## Condition Summary

| mode | N | sim>=t | update sec | eval sec | avg anchor rank delta | avg anchor score delta | avg neighbor rank delta | avg neighbor score delta | avg same-scenario outside-neighbors rank delta | avg same-scenario outside-neighbors score delta | avg cross-scenario rank delta | avg cross-scenario score delta | selected-scenario most_rel before | selected-scenario most_rel after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hard-assigned-local-cutoff` | `2000` | 0.8 | 0.8 | 6.4 | -0.833 | -0.2141 | -0.461 | -0.2101 | -0.374 | -0.2576 | -1.594 | -0.0330 | 2.952 | 2.826 |

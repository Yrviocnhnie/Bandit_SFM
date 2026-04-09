# Feedback Propagation Experiment

- artifact_dir: `artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split`
- device: `cpu`
- n_values: `1,5,10,20,50,100`

## Locked Feedback Items

| scenario | feedback | target_position | target_action_id | anchor_episode_id | baseline_top3 |
| --- | --- | ---: | --- | --- | --- |
| `ARRIVE_OFFICE` | `like` | 1 | `O_SHOW_SCHEDULE` | `arrive_office_u000002` | `O_SHOW_SCHEDULE , O_SHOW_TODAY_TODO , R_PLAN_DAY_OVER_COFFEE` |
| `OFFICE_LUNCH_OUT` | `dislike` | 1 | `O_SHOW_NEARBY_OPTIONS` | `office_lunch_out_a000002` | `O_SHOW_NEARBY_OPTIONS , R_MEAL_BREAK , O_SHOW_PAYMENT_QR` |
| `OFFICE_WORKING` | `like` | 2 | `O_SHOW_SCHEDULE` | `office_afternoon_u000008` | `O_SHOW_TODAY_TODO , O_SHOW_SCHEDULE , R_WORK_BREAK_AND_STRETCH` |
| `LEAVE_OFFICE` | `dislike` | 2 | `R_CLOCK_OUT_BEFORE_LEAVING` | `leave_office_a000009` | `O_SHOW_WEATHER , R_CLOCK_OUT_BEFORE_LEAVING , O_SHOW_COMMUTE_TRAFFIC` |

## Condition Summary

| mode | N | avg anchor rank delta | avg same-scenario rank delta | avg cross-scenario rank delta | selected-scenario most_rel before | selected-scenario most_rel after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single` | `1` | 0.000 | -0.023 | -0.001 | 2.976 | 2.976 |
| `global-latent-nearest` | `1` | 0.000 | -0.025 | -0.002 | 2.976 | 2.976 |
| `global-latent-nearest` | `5` | 0.000 | -0.124 | -0.006 | 2.976 | 2.976 |
| `global-latent-nearest` | `10` | -0.250 | -0.240 | -0.008 | 2.976 | 2.976 |
| `global-latent-nearest` | `20` | -0.500 | -0.364 | -0.016 | 2.976 | 2.976 |
| `global-latent-nearest` | `50` | -0.750 | -0.413 | -0.039 | 2.976 | 2.976 |
| `global-latent-nearest` | `100` | -0.750 | -0.417 | -0.076 | 2.976 | 2.976 |
| `same-scenario-nearest` | `1` | 0.000 | -0.026 | -0.002 | 2.976 | 2.976 |
| `same-scenario-nearest` | `5` | 0.000 | -0.124 | -0.004 | 2.976 | 2.976 |
| `same-scenario-nearest` | `10` | -0.250 | -0.241 | -0.008 | 2.976 | 2.976 |
| `same-scenario-nearest` | `20` | -0.500 | -0.364 | -0.015 | 2.976 | 2.976 |
| `same-scenario-nearest` | `50` | -0.750 | -0.413 | -0.040 | 2.976 | 2.976 |
| `same-scenario-nearest` | `100` | -0.750 | -0.416 | -0.074 | 2.976 | 2.976 |
| `entire-scenario-all` | `all` | -17.000 | -9.435 | -0.297 | 2.976 | 2.862 |

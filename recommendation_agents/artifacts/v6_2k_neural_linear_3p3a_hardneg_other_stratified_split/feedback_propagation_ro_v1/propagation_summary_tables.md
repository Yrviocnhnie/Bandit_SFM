# Feedback Propagation Summary

- source: `artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_propagation_ro_v1/results.json`

## Per-Feedback-Item Table

### `ARRIVE_OFFICE`

- feedback: `like`
- target_action_id: `O_SHOW_SCHEDULE`
- anchor_episode_id: `arrive_office_u000002`

| mode | N | eff N | anchor rank before | anchor rank after | anchor aligned delta | same-scenario aligned delta | cross-scenario aligned delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single` | `1` | 1 | 1 | 1 | 0.000 | 0.000 | -0.000 |
| `global-latent-nearest` | `1` | 1 | 1 | 1 | 0.000 | -0.000 | -0.002 |
| `global-latent-nearest` | `5` | 5 | 1 | 1 | 0.000 | -0.001 | -0.001 |
| `global-latent-nearest` | `10` | 10 | 1 | 1 | 0.000 | -0.002 | -0.001 |
| `global-latent-nearest` | `20` | 20 | 1 | 1 | 0.000 | -0.003 | 0.000 |
| `global-latent-nearest` | `50` | 50 | 1 | 1 | 0.000 | -0.008 | 0.002 |
| `global-latent-nearest` | `100` | 100 | 1 | 1 | 0.000 | -0.015 | 0.000 |
| `same-scenario-nearest` | `1` | 1 | 1 | 1 | 0.000 | 0.000 | -0.000 |
| `same-scenario-nearest` | `5` | 5 | 1 | 1 | 0.000 | -0.001 | 0.001 |
| `same-scenario-nearest` | `10` | 10 | 1 | 1 | 0.000 | -0.002 | 0.002 |
| `same-scenario-nearest` | `20` | 20 | 1 | 1 | 0.000 | -0.003 | 0.002 |
| `same-scenario-nearest` | `50` | 50 | 1 | 1 | 0.000 | -0.008 | 0.000 |
| `same-scenario-nearest` | `100` | 100 | 1 | 1 | 0.000 | -0.013 | 0.004 |
| `entire-scenario-all` | `all` | 20736 | 1 | 2 | -1.000 | -1.384 | 0.546 |

### `OFFICE_LUNCH_OUT`

- feedback: `dislike`
- target_action_id: `O_SHOW_NEARBY_OPTIONS`
- anchor_episode_id: `office_lunch_out_a000002`

| mode | N | eff N | anchor rank before | anchor rank after | anchor aligned delta | same-scenario aligned delta | cross-scenario aligned delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single` | `1` | 1 | 1 | 1 | 0.000 | 0.050 | -0.001 |
| `global-latent-nearest` | `1` | 1 | 1 | 1 | 0.000 | 0.053 | 0.001 |
| `global-latent-nearest` | `5` | 5 | 1 | 1 | 0.000 | 0.278 | 0.003 |
| `global-latent-nearest` | `10` | 10 | 1 | 1 | 0.000 | 0.594 | 0.006 |
| `global-latent-nearest` | `20` | 20 | 1 | 2 | 1.000 | 0.987 | 0.010 |
| `global-latent-nearest` | `50` | 50 | 1 | 3 | 2.000 | 1.138 | 0.027 |
| `global-latent-nearest` | `100` | 100 | 1 | 3 | 2.000 | 1.138 | 0.068 |
| `same-scenario-nearest` | `1` | 1 | 1 | 1 | 0.000 | 0.053 | 0.001 |
| `same-scenario-nearest` | `5` | 5 | 1 | 1 | 0.000 | 0.278 | 0.003 |
| `same-scenario-nearest` | `10` | 10 | 1 | 1 | 0.000 | 0.594 | 0.006 |
| `same-scenario-nearest` | `20` | 20 | 1 | 2 | 1.000 | 0.987 | 0.010 |
| `same-scenario-nearest` | `50` | 50 | 1 | 3 | 2.000 | 1.138 | 0.027 |
| `same-scenario-nearest` | `100` | 100 | 1 | 3 | 2.000 | 1.138 | 0.068 |
| `entire-scenario-all` | `all` | 1600 | 1 | 43 | 42.000 | 11.366 | 0.807 |

### `OFFICE_WORKING`

- feedback: `like`
- target_action_id: `O_SHOW_SCHEDULE`
- anchor_episode_id: `office_afternoon_u000008`

| mode | N | eff N | anchor rank before | anchor rank after | anchor aligned delta | same-scenario aligned delta | cross-scenario aligned delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single` | `1` | 1 | 2 | 2 | 0.000 | -0.002 | -0.001 |
| `global-latent-nearest` | `1` | 1 | 2 | 2 | 0.000 | 0.002 | -0.001 |
| `global-latent-nearest` | `5` | 5 | 2 | 2 | 0.000 | -0.005 | -0.004 |
| `global-latent-nearest` | `10` | 10 | 2 | 2 | 0.000 | -0.005 | -0.000 |
| `global-latent-nearest` | `20` | 20 | 2 | 2 | 0.000 | -0.007 | -0.002 |
| `global-latent-nearest` | `50` | 50 | 2 | 2 | 0.000 | -0.011 | -0.002 |
| `global-latent-nearest` | `100` | 100 | 2 | 2 | 0.000 | -0.020 | -0.005 |
| `same-scenario-nearest` | `1` | 1 | 2 | 2 | 0.000 | -0.002 | -0.002 |
| `same-scenario-nearest` | `5` | 5 | 2 | 2 | 0.000 | -0.002 | -0.001 |
| `same-scenario-nearest` | `10` | 10 | 2 | 2 | 0.000 | -0.007 | -0.002 |
| `same-scenario-nearest` | `20` | 20 | 2 | 2 | 0.000 | -0.007 | 0.002 |
| `same-scenario-nearest` | `50` | 50 | 2 | 2 | 0.000 | -0.011 | -0.004 |
| `same-scenario-nearest` | `100` | 100 | 2 | 2 | 0.000 | -0.018 | -0.003 |
| `entire-scenario-all` | `all` | 1766 | 2 | 2 | 0.000 | -0.077 | 0.390 |

### `LEAVE_OFFICE`

- feedback: `dislike`
- target_action_id: `R_CLOCK_OUT_BEFORE_LEAVING`
- anchor_episode_id: `leave_office_a000009`

| mode | N | eff N | anchor rank before | anchor rank after | anchor aligned delta | same-scenario aligned delta | cross-scenario aligned delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single` | `1` | 1 | 2 | 2 | 0.000 | 0.040 | 0.003 |
| `global-latent-nearest` | `1` | 1 | 2 | 2 | 0.000 | 0.048 | 0.003 |
| `global-latent-nearest` | `5` | 5 | 2 | 2 | 0.000 | 0.213 | 0.014 |
| `global-latent-nearest` | `10` | 10 | 2 | 3 | 1.000 | 0.361 | 0.027 |
| `global-latent-nearest` | `20` | 20 | 2 | 3 | 1.000 | 0.459 | 0.053 |
| `global-latent-nearest` | `50` | 50 | 2 | 3 | 1.000 | 0.494 | 0.129 |
| `global-latent-nearest` | `100` | 100 | 2 | 3 | 1.000 | 0.496 | 0.230 |
| `same-scenario-nearest` | `1` | 1 | 2 | 2 | 0.000 | 0.048 | 0.003 |
| `same-scenario-nearest` | `5` | 5 | 2 | 2 | 0.000 | 0.213 | 0.015 |
| `same-scenario-nearest` | `10` | 10 | 2 | 3 | 1.000 | 0.361 | 0.027 |
| `same-scenario-nearest` | `20` | 20 | 2 | 3 | 1.000 | 0.459 | 0.053 |
| `same-scenario-nearest` | `50` | 50 | 2 | 3 | 1.000 | 0.494 | 0.129 |
| `same-scenario-nearest` | `100` | 100 | 2 | 3 | 1.000 | 0.496 | 0.230 |
| `entire-scenario-all` | `all` | 1600 | 2 | 27 | 25.000 | 24.912 | 1.317 |

## Aligned Aggregate Table

Aligned means:
- `like`: higher score if rank moves forward
- `dislike`: higher score if rank moves backward
- so positive values are always better aligned with user feedback

| mode | N | avg aligned anchor delta | avg aligned same-scenario delta | avg aligned cross-scenario delta | selected most_rel before | selected most_rel after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single` | `1` | 0.000 | 0.022 | 0.000 | 2.976 | 2.976 |
| `global-latent-nearest` | `1` | 0.000 | 0.026 | 0.000 | 2.976 | 2.976 |
| `global-latent-nearest` | `5` | 0.000 | 0.121 | 0.003 | 2.976 | 2.976 |
| `global-latent-nearest` | `10` | 0.250 | 0.237 | 0.008 | 2.976 | 2.976 |
| `global-latent-nearest` | `20` | 0.500 | 0.359 | 0.015 | 2.976 | 2.976 |
| `global-latent-nearest` | `50` | 0.750 | 0.403 | 0.039 | 2.976 | 2.976 |
| `global-latent-nearest` | `100` | 0.750 | 0.400 | 0.073 | 2.976 | 2.976 |
| `same-scenario-nearest` | `1` | 0.000 | 0.025 | 0.000 | 2.976 | 2.976 |
| `same-scenario-nearest` | `5` | 0.000 | 0.122 | 0.004 | 2.976 | 2.976 |
| `same-scenario-nearest` | `10` | 0.250 | 0.237 | 0.008 | 2.976 | 2.976 |
| `same-scenario-nearest` | `20` | 0.500 | 0.359 | 0.017 | 2.976 | 2.976 |
| `same-scenario-nearest` | `50` | 0.750 | 0.403 | 0.038 | 2.976 | 2.976 |
| `same-scenario-nearest` | `100` | 0.750 | 0.401 | 0.075 | 2.976 | 2.976 |
| `entire-scenario-all` | `all` | 16.500 | 8.704 | 0.765 | 2.976 | 2.862 |

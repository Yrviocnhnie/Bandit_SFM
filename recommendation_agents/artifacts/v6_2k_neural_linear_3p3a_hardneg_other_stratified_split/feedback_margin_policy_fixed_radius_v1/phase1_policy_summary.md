# Margin-Aware Fixed-Radius Policy v1

This run applies six simulated user feedback items together on the frozen NeuralEncoder LinUCB R/O model.

## Policy

- propagation mode: `hard-assigned-local-cutoff`
- fixed radius: `N=2000`
- similarity threshold: `0.80`
- reward policy: `margin-aware-fixed-radius-v1`
- small gap: like `+2`, dislike `-0.1`
- medium gap: like `+5`, dislike `-1`
- large gap: like `+10`, dislike `-10`
- boundary cases: like rank1 uses conservative `+2`; dislike last-rank would use conservative `-1`

## Feedback Results

| feedback_id | scenario | feedback | target action | gap/bin | reward | effective N | anchor rank | score delta | top3 before | top3 after |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| `arrive_office_like_rank1` | `ARRIVE_OFFICE` | `like` | `O_SHOW_SCHEDULE` | 0.0000 / `boundary` | 2.0 | 1566 | 1 -> 1 | 0.266 | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_PLAN_DAY_OVER_COFFEE` | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_PLAN_DAY_OVER_COFFEE` |
| `office_lunch_out_dislike_rank1` | `OFFICE_LUNCH_OUT` | `dislike` | `O_SHOW_NEARBY_OPTIONS` | 0.0211 / `small` | -0.1 | 1600 | 1 -> 3 | -0.571 | `O_SHOW_NEARBY_OPTIONS`<br>`R_MEAL_BREAK`<br>`O_SHOW_PAYMENT_QR` | `R_MEAL_BREAK`<br>`O_SHOW_PAYMENT_QR`<br>`O_SHOW_NEARBY_OPTIONS` |
| `office_working_like_rank2` | `OFFICE_WORKING` | `like` | `O_SHOW_SCHEDULE` | 0.0006 / `small` | 2.0 | 1640 | 2 -> 1 | 0.255 | `O_SHOW_TODAY_TODO`<br>`O_SHOW_SCHEDULE`<br>`R_WORK_BREAK_AND_STRETCH` | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_WORK_BREAK_AND_STRETCH` |
| `leave_office_dislike_rank2` | `LEAVE_OFFICE` | `dislike` | `R_CLOCK_OUT_BEFORE_LEAVING` | 0.0079 / `small` | -0.1 | 1600 | 2 -> 3 | -0.553 | `O_SHOW_WEATHER`<br>`R_CLOCK_OUT_BEFORE_LEAVING`<br>`O_SHOW_COMMUTE_TRAFFIC` | `O_SHOW_WEATHER`<br>`O_SHOW_COMMUTE_TRAFFIC`<br>`R_CLOCK_OUT_BEFORE_LEAVING` |
| `cafe_stay_like_rank3` | `CAFE_STAY` | `like` | `O_SHOW_PAYMENT_QR` | 0.0224 / `small` | 2.0 | 2000 | 3 -> 1 | 0.365 | `O_SHOW_SCHEDULE`<br>`R_ENJOY_LEISURE_MOMENT`<br>`O_SHOW_PAYMENT_QR` | `O_SHOW_PAYMENT_QR`<br>`O_SHOW_SCHEDULE`<br>`R_ENJOY_LEISURE_MOMENT` |
| `office_to_cafe_dislike_rank3` | `OFFICE_TO_CAFE` | `dislike` | `O_SHOW_NEARBY_OPTIONS` | 0.9790 / `large` | -10.0 | 2000 | 3 -> 8 | -1.047 | `R_ENJOY_LEISURE_MOMENT`<br>`O_SHOW_PAYMENT_QR`<br>`O_SHOW_NEARBY_OPTIONS` | `R_ENJOY_LEISURE_MOMENT`<br>`O_SHOW_PAYMENT_QR`<br>`O_SHOW_WEATHER` |

## Propagation Metrics

| feedback_id | same-scenario avg rank delta | same-scenario avg score delta | cross-scenario avg rank delta | cross-scenario avg score delta |
| --- | ---: | ---: | ---: | ---: |
| `arrive_office_like_rank1` | 0.146 | 0.024 | -0.155 | 0.0124 |
| `office_lunch_out_dislike_rank1` | -1.138 | -0.554 | -4.162 | -0.1293 |
| `office_working_like_rank2` | 0.526 | 0.231 | -0.178 | 0.0095 |
| `leave_office_dislike_rank2` | -0.496 | -0.545 | -0.742 | -0.0001 |
| `cafe_stay_like_rank3` | 1.345 | 0.308 | 0.180 | 0.0014 |
| `office_to_cafe_dislike_rank3` | -2.626 | -1.010 | -4.505 | -0.0920 |

## Aggregate

- update time: `0.81s`; evaluation time: `6.14s`; total condition time: `6.95s`
- cache hits: train=`True`, test=`True`, neighbors=`True`, baseline=`True`
- feedback cache size: about `45 MB`
- final policy output size: about `92 KB`
- effective training contexts per feedback item: `[1566, 1600, 1640, 1600, 2000, 2000]`
- average effective training contexts per feedback item: `1734.3`
- total replay/update points: `10406`
- selected-scenario offline most-relevant@3: `2.952` -> `2.826`
- selected-scenario acceptable@3: `2.972` -> `2.972`
- selected-scenario irrelevant@3: `0.000` -> `0.000`

Note: offline most-relevant@3 can decrease when the user explicitly dislikes an offline most-relevant action. In that case the more relevant success metric is target-action movement and replacement quality.

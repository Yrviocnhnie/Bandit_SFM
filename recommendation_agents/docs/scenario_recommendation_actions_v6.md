# v6 Global Shared Action Space

## 中文说明

这版 v6 延续 v5 的共享 global action space 设计，并在 scenario defaults 中额外补充了用于离线训练展开的辅助 action 映射。

核心变化：
- legacy `R/O` action 总数：**167**
- global `R/O` action 总数：**46**
- 其中 global `R`：**26**
- 其中 global `O`：**20**
- scenario 总数：**65**
- 每个 scenario 现在都给出 **>=2 个默认 R/O actions**，按 common sense 排序
- app 仍然使用固定的 **10 个 app categories**，但每个 scenario 现在给出 **>=3 个默认 app categories**，按适配度排序

这次重构的重点：
- **O 怎么合并**：优先按 operation 合并；如果需要参数，这些参数必须能由当前已有 feature 直接 resolve，模型本身只推荐 action
- **R 怎么合并**：优先按用户真实意图合并，比如“进入工作节奏”“工作中休息活动”“运动后恢复”“查看行程/预约信息”

输出文件：
- `outputs/global_action_space_details.json`：完整 action space 文件，里面同时包含 global R/O action space 和每个 scenario 的映射、默认 action 顺序、默认 app 顺序
- `outputs/global_action_space.json`：更轻量的 action space 文件，不包含 old-new mapping，只保留 global action space 和每个 scenario 的默认 action / 默认 app
- 原始 legacy 数据保留在 `references/scenario_actions.json`

## Merge Principles

- O 优先按 operation 合并；如果需要参数，则这些参数必须能由当前已有 feature 直接解析，RL/UCB 只负责推荐 action。
- R 优先按用户意图合并；如果只是文案、场景措辞不同，则收敛到统一提醒语义。
- scenario 层只保留 action 引用、触发条件和默认顺序，不携带 action 参数；参数在执行层由当前 feature 直接决定，具体定义见输出 JSON。
- 每个 scenario 至少保留 2 个默认 R/O action，并为 app category 给出至少 3 个按 common sense 排序的默认项。

## Global Reminder Space

| global R actionId | message | merged legacy count | 合并说明 |
| --- | --- | --- | --- |
| `R_AFTER_MEAL_WALK` | 饭后出去走一走，会比较舒服。 | 1 | 保留饭后散步的独立提醒。 |
| `R_CALENDAR_QUICK_LOOK` | 今天看起来有安排，快速看一眼会更有数。 | 1 | 保留“今天有安排，先快速看一眼”的轻提醒。 |
| `R_CAPTURE_MEETING_FOLLOW_UPS` | 一边专注讨论，一边把后续要跟进的点顺手记下来。 | 1 | 保留会议中记录 follow-up 的提醒。 |
| `R_CHECK_TRIP_OR_BOOKING_INFO` | 建议先确认这次行程或预约的关键信息，避免临近时手忙脚乱。 | 7 | 把交通出发、酒店、电影、医院、网约车等“先确认关键信息”的提醒合并。 |
| `R_CLOCK_OUT_BEFORE_LEAVING` | 离开前别忘了把该处理的离场动作做完，比如打卡。 | 1 | 保留离开办公室前的明确动作提醒。 |
| `R_COMMUTE_STEADY_AND_SAFE` | 路上稳一点，不用太赶，安全最重要。 | 2 | 合并通勤中、骑行中等“注意节奏和安全”的提醒。 |
| `R_DEEP_WORK_WINDOW` | 现在这段时间挺适合专注做点重要的事。 | 3 | 合并无会议、安静学校、安静咖啡馆等“适合专注”的提醒。 |
| `R_DELIVERY_PICKUP_REMINDER` | 有一件东西待处理，方便的话顺手去取一下。 | 1 | 保留快递/取件的明确提醒。 |
| `R_DEPART_EARLY_FOR_OFFSITE` | 这次需要出行，提前一点出发会更从容。 | 1 | 保留需要出行去开会/赴约时的提前出发提醒。 |
| `R_ENJOY_LEISURE_MOMENT` | 这段时间适合放松一点，享受当下就好。 | 7 | 合并购物、餐饮、咖啡、散步、周末外出等“享受过程”的轻量提醒。 |
| `R_HOME_RELAX_AND_RESET` | 可以让自己慢下来，补点水、放松一下，再决定接下来的安排。 | 6 | 合并回家后、居家下午、居家傍晚、安静在家等放松类提醒。 |
| `R_LONG_DRIVE_REST` | 连续驾驶已经有一段时间了，找个合适的地方休息一下。 | 1 | 保留连续驾驶后需要休息的强提醒。 |
| `R_MEAL_BREAK` | 差不多该吃点东西了，顺便休息一下会更舒服。 | 2 | 合并午餐时间、办公室午间外出等用餐提醒。 |
| `R_NOISY_ENV_STAY_SAFE` | 周围有点嘈杂，记得留意安全和重要物品。 | 2 | 合并人多嘈杂、深夜嘈杂等安全提醒。 |
| `R_OPEN_CURTAINS_OR_LIGHT` | 环境已经暗了挺久了，可以考虑开灯、拉窗帘，调整一下状态。 | 1 | 保留暗环境持续过久时的环境调整提醒。 |
| `R_OUTDOOR_HYDRATE_AND_REST` | 户外活动别忘了补水，累了就找个地方稍微歇一会儿。 | 2 | 合并户外跑步、久走等场景里的补水与休息提醒。 |
| `R_PLAN_DAY_OVER_COFFEE` | 借这几分钟想一下今天最重要的事，慢慢进入节奏。 | 2 | 把“喝咖啡”“开始一天”“顺手想清重点”收敛为同一个轻量启动提醒。 |
| `R_POST_WORKOUT_RECOVERY` | 运动后记得拉伸、补水，给身体一点恢复时间。 | 4 | 合并健身后拉伸、长时间训练后恢复、运动后补水等恢复提醒。 |
| `R_PREPARE_FOR_MEETING` | 会议快开始了，先简单准备一下会更从容。 | 1 | 保留会议即将开始前的准备提醒。 |
| `R_SLEEP_SOON_AND_CHARGE` | 时间已经比较晚了，尽量准备休息，也别忘了把手机和自己都安顿好。 | 3 | 合并深夜在家、晚归到家、深夜躺着玩手机等“该休息了”的提醒。 |
| `R_WAITING_TIME_USEFUL` | 这段等待时间可以利用起来，看看安排或处理一点小事。 | 1 | 合并教育场景里等待时可看看安排/处理待办的提醒。 |
| `R_WEEKEND_OR_MORNING_GREETING` | 今天可以舒服一点进入状态，不用太赶。 | 2 | 合并起床、周末早晨等“慢慢进入状态”的轻提醒。 |
| `R_WORK_BREAK_AND_STRETCH` | 已经持续一段时间了，起来活动一下，也别忘了补水和放松眼睛。 | 7 | 合并办公室、居家办公、长时间坐着、长时间专注等场景下的休息、喝水、活动提醒。 |
| `R_WORK_START_SETTLE_IN` | 先稳一下节奏，再开始接下来的工作会更顺。 | 2 | 覆盖到公司后、午饭回办公室后等“重新进入工作状态”的提醒。 |
| `R_WORKOUT_PREP` | 开始训练前先热热身、补点水，会更稳一些。 | 3 | 合并晨练热身、组间休息后准备继续训练、去健身前补水等准备动作。 |
| `R_WRAP_UP_AND_REST_AFTER_WORK` | 时间已经不早了，尽量收尾当前事情，给自己留一点休息空间。 | 2 | 合并傍晚留守办公室、深夜加班等“该收尾了”的提醒。 |

## Global Operation Space

| global O actionId | message | merged legacy count | 合并说明 | operation |
| --- | --- | --- | --- | --- |
| `O_ENABLE_VIBRATION` | 开启振动提醒。 | 3 | 合并嘈杂环境下的振动提醒操作。 | `enable_vibration` |
| `O_INCREASE_NOTIFICATION_VOLUME` | 调高通知音量。 | 3 | 合并 `increase_volume` 与 `increase_volume_to_max`，强度由 JSON 中的参数定义决定。 | `increase_volume` |
| `O_OPEN_NOTES_QUICK_ENTRY` | 打开快速记录入口。 | 1 | 保留会议中打开快捷记录入口的操作。 | `open_notes_quick_entry` |
| `O_PROMPT_ADD_FREQUENT_PLACE` | 提示把当前地点添加为常用地点。 | 1 | 保留未知地点停留过久时的建点操作。 | `prompt_add_frequent_place` |
| `O_SHOW_BOOKING_DETAILS` | 查看这次预约或订单的详情。 | 3 | 合并酒店、电影、医院等预约或订单详情查看操作。 | `show_booking_details` |
| `O_SHOW_COMMUTE_TRAFFIC` | 查看路况和预计到达时间。 | 3 | 合并上下班与离开办公室后的通勤路况查看操作。 | `show_commute_traffic` |
| `O_SHOW_MEETING_DETAILS` | 查看会议的时间、地点和相关信息。 | 2 | 合并会议即将开始、异地会议中的会议信息查看操作。 | `show_meeting_details` |
| `O_SHOW_NEARBY_OPTIONS` | 查看附近有哪些合适的去处。 | 5 | 合并附近地点、附近餐厅、附近服务区这类 nearby 查询操作。 | `show_nearby_places` |
| `O_SHOW_PAYMENT_QR` | 先把付款码准备好。 | 6 | 合并餐饮、购物、咖啡、电影等付款码相关操作。 | `show_payment_qr` |
| `O_SHOW_PICKUP_DETAILS` | 查看取件相关信息。 | 1 | 保留取件信息查看操作。 | `show_pickup_details` |
| `O_SHOW_SCHEDULE` | 查看日程安排。 | 17 | 合并查看今日日程、明日安排与日程概览。 | `show_schedule` |
| `O_SHOW_TODAY_TODO` | 查看今天的待办事项。 | 14 | 把所有查看今日待办的场景合并成一个共享操作。 | `show_today_todo_list` |
| `O_SHOW_TRAVEL_INFO` | 查看当前行程的关键信息。 | 5 | 合并交通枢纽、列车、航班、网约车、异地会议等行程详情查看操作。 | `show_travel_information` |
| `O_SHOW_WALKING_ROUTE` | 查看附近的步行路线或返程路线。 | 3 | 合并公园散步、饭后散步、周末散步的路线查看操作。 | `show_walking_route` |
| `O_SHOW_WEATHER` | 查看天气情况。 | 7 | 合并当前天气与今天天气查看操作。 | `show_weather` |
| `O_START_ACTIVITY_TRACKING` | 开始记录这次活动。 | 7 | 合并晨练、健身、跑步、骑行、步行等 tracking 操作。 | `start_activity_tracking` |
| `O_START_CONTEXT_TIMER` | 开启一个合适的计时器。 | 7 | 合并通用计时、休息计时、恢复计时操作。 | `start_context_timer` |
| `O_TURN_ON_DND` | 开启勿扰模式。 | 3 | 合并会议、睡前等场景中的勿扰模式操作。 | `turn_on_dnd` |
| `O_TURN_ON_EYE_COMFORT_MODE` | 开启护眼模式。 | 8 | 把所有护眼模式相关操作合并为一个共享 action。 | `turn_on_eye_comfort_mode` |
| `O_TURN_ON_SILENT_MODE` | 开启静音模式。 | 2 | 合并学校、安静咖啡馆中的静音操作。 | `turn_on_silent_mode` |

## Scenario Defaults

在 v6 中，`Scenario Defaults` 统一整理成固定的 `3+3+2` 结构，分别为：

- `most relevant 3`：最相关、最希望进入 UI top-3 的 3 个候选
- `other plausible 3`：不是 top-3，但仍然可能相关、可作为补充正样本的 3 个候选
- `irrelevant 2`：明显不相关、可作为负样本候选的 2 个候选
- `extra hard negative R/O actionIds`：基于 train split mining 后人工确认的新 R/O hard negatives

R/O 和 App 都采用同样的结构。这里仍然只定义 action/category 映射，不在文档中直接绑定 reward；reward 会在后续数据复制与训练配置中决定。

| scenarioId | scenarioNameZh | most relevant 3 R/O actionIds | other plausible 3 R/O actionIds | irrelevant 2 R/O actionIds | extra hard negative R/O actionIds | most relevant 3 app categories | other plausible 3 app categories | irrelevant 2 app categories |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ARRIVE_OFFICE` | 到达办公室 | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_PLAN_DAY_OVER_COFFEE` | `R_WORK_START_SETTLE_IN`<br>`R_DEEP_WORK_WINDOW`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` | `O_START_ACTIVITY_TRACKING`<br>`R_HOME_RELAX_AND_RESET`<br>`O_SHOW_NEARBY_OPTIONS` | `productivity`<br>`news`<br>`music` | `reading`<br>`social`<br>`health` | `shopping`<br>`game` |
| `OFFICE_LUNCH_OUT` | 办公室午间外出 | `O_SHOW_PAYMENT_QR`<br>`R_MEAL_BREAK`<br>`O_SHOW_NEARBY_OPTIONS` | `R_ENJOY_LEISURE_MOMENT`<br>`O_SHOW_WEATHER`<br>`O_SHOW_BOOKING_DETAILS` | `O_START_ACTIVITY_TRACKING`<br>`O_TURN_ON_DND` |  | `shopping`<br>`social`<br>`navigation` | `music`<br>`news`<br>`reading` | `health`<br>`productivity` |
| `OFFICE_WORKING` | 办公室工作 | `O_SHOW_TODAY_TODO`<br>`O_SHOW_SCHEDULE`<br>`R_WORK_BREAK_AND_STRETCH` | `R_DEEP_WORK_WINDOW`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`R_WORK_START_SETTLE_IN` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` |  | `productivity`<br>`music`<br>`news` | `reading`<br>`social`<br>`health` | `shopping`<br>`game` |
| `LEAVE_OFFICE` | 离开办公室 | `O_SHOW_COMMUTE_TRAFFIC`<br>`O_SHOW_WEATHER`<br>`R_CLOCK_OUT_BEFORE_LEAVING` | `O_SHOW_SCHEDULE`<br>`O_SHOW_TRAVEL_INFO`<br>`R_CHECK_TRIP_OR_BOOKING_INFO` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` | `O_SHOW_NEARBY_OPTIONS` | `navigation`<br>`music`<br>`social` | `shopping`<br>`productivity`<br>`reading` | `game`<br>`health` |
| `WEEKDAY_HOME_DAY` | 工作日居家白天 | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_WORK_BREAK_AND_STRETCH` | `R_DEEP_WORK_WINDOW`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`R_WORK_START_SETTLE_IN` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` | `R_HOME_RELAX_AND_RESET`<br>`O_SHOW_NEARBY_OPTIONS` | `productivity`<br>`reading`<br>`music` | `social`<br>`health`<br>`navigation` | `shopping`<br>`game` |
| `ARRIVE_TRANSIT_HUB` | 到达交通枢纽 | `O_SHOW_TRAVEL_INFO`<br>`R_CHECK_TRIP_OR_BOOKING_INFO`<br>`O_SHOW_BOOKING_DETAILS` | `O_SHOW_COMMUTE_TRAFFIC`<br>`O_SHOW_WEATHER`<br>`O_SHOW_SCHEDULE` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` |  | `navigation`<br>`productivity`<br>`news` | `social`<br>`shopping`<br>`reading` | `game`<br>`health` |
| `LATE_NIGHT_OVERTIME` | 深夜加班 | `R_WRAP_UP_AND_REST_AFTER_WORK`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`O_SHOW_TODAY_TODO` | `R_DEEP_WORK_WINDOW`<br>`R_WORK_START_SETTLE_IN`<br>`O_SHOW_SCHEDULE` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` | `R_HOME_RELAX_AND_RESET` | `navigation`<br>`productivity`<br>`music` | `reading`<br>`social`<br>`health` | `shopping`<br>`game` |
| `WEEKEND_OVERTIME` | 周末加班 | `R_WORK_BREAK_AND_STRETCH`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`O_SHOW_TODAY_TODO` | `R_DEEP_WORK_WINDOW`<br>`R_WORK_START_SETTLE_IN`<br>`O_SHOW_SCHEDULE` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` |  | `productivity`<br>`music`<br>`news` | `reading`<br>`social`<br>`health` | `shopping`<br>`game` |
| `EVENING_AT_OFFICE` | 傍晚留守办公室 | `O_SHOW_COMMUTE_TRAFFIC`<br>`O_SHOW_TODAY_TODO`<br>`R_WRAP_UP_AND_REST_AFTER_WORK` | `O_TURN_ON_EYE_COMFORT_MODE`<br>`R_DEEP_WORK_WINDOW`<br>`R_WORK_START_SETTLE_IN` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` |  | `social`<br>`navigation`<br>`productivity` | `reading`<br>`health`<br>`entertainment` | `shopping`<br>`game` |
| `MORNING_EXERCISE` | 晨练 | `O_START_ACTIVITY_TRACKING`<br>`O_START_CONTEXT_TIMER`<br>`R_WORKOUT_PREP` | `O_SHOW_WEATHER`<br>`R_POST_WORKOUT_RECOVERY`<br>`R_OUTDOOR_HYDRATE_AND_REST` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` |  | `health`<br>`music`<br>`productivity` | `news`<br>`reading`<br>`entertainment` | `shopping`<br>`social` |
| `SHOPPING` | 购物 | `O_SHOW_PAYMENT_QR`<br>`R_ENJOY_LEISURE_MOMENT`<br>`O_SHOW_NEARBY_OPTIONS` | `R_MEAL_BREAK`<br>`O_SHOW_WEATHER`<br>`O_SHOW_BOOKING_DETAILS` | `O_TURN_ON_DND`<br>`O_OPEN_NOTES_QUICK_ENTRY` | `O_START_ACTIVITY_TRACKING` | `shopping`<br>`social`<br>`navigation` | `music`<br>`news`<br>`reading` | `health`<br>`productivity` |
| `HOME_EVENING` | 居家傍晚 | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_HOME_RELAX_AND_RESET` | `O_TURN_ON_EYE_COMFORT_MODE`<br>`R_SLEEP_SOON_AND_CHARGE`<br>`O_SHOW_WEATHER` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` |  | `entertainment`<br>`music`<br>`reading` | `social`<br>`news`<br>`productivity` | `navigation`<br>`shopping` |
| `ARRIVE_DINING` | 到达餐厅/咖啡厅 | `O_SHOW_PAYMENT_QR`<br>`R_ENJOY_LEISURE_MOMENT`<br>`O_SHOW_NEARBY_OPTIONS` | `R_MEAL_BREAK`<br>`O_SHOW_WEATHER`<br>`O_SHOW_BOOKING_DETAILS` | `O_TURN_ON_DND`<br>`O_OPEN_NOTES_QUICK_ENTRY` | `O_START_ACTIVITY_TRACKING` | `shopping`<br>`social`<br>`navigation` | `music`<br>`news`<br>`reading` | `health`<br>`productivity` |
| `HOME_LATE_NIGHT` | 居家深夜 | `R_SLEEP_SOON_AND_CHARGE`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`O_SHOW_SCHEDULE` | `O_TURN_ON_DND`<br>`R_HOME_RELAX_AND_RESET`<br>`O_SHOW_TODAY_TODO` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` |  | `music`<br>`reading`<br>`health` | `social`<br>`news`<br>`productivity` | `navigation`<br>`shopping` |
| `WEEKEND_MORNING` | 周末早晨 | `O_SHOW_WEATHER`<br>`O_SHOW_SCHEDULE`<br>`R_WEEKEND_OR_MORNING_GREETING` | `O_SHOW_TODAY_TODO`<br>`R_HOME_RELAX_AND_RESET`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `O_TURN_ON_DND`<br>`O_START_ACTIVITY_TRACKING` | `R_ENJOY_LEISURE_MOMENT`<br>`O_PROMPT_ADD_FREQUENT_PLACE` | `news`<br>`entertainment`<br>`game` | `social`<br>`productivity`<br>`health` | `navigation`<br>`shopping` |
| `WEEKEND_OUTING` | 周末外出 | `O_SHOW_WEATHER`<br>`O_SHOW_NEARBY_OPTIONS`<br>`R_ENJOY_LEISURE_MOMENT` | `O_SHOW_PAYMENT_QR`<br>`R_MEAL_BREAK`<br>`O_SHOW_BOOKING_DETAILS` | `O_TURN_ON_DND`<br>`O_OPEN_NOTES_QUICK_ENTRY` | `O_SHOW_WALKING_ROUTE`<br>`O_START_ACTIVITY_TRACKING` | `navigation`<br>`social`<br>`entertainment` | `music`<br>`news`<br>`reading` | `health`<br>`productivity` |
| `LATE_RETURN_HOME` | 深夜回家 | `R_SLEEP_SOON_AND_CHARGE`<br>`O_TURN_ON_DND`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `R_HOME_RELAX_AND_RESET`<br>`O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` |  | `music`<br>`health`<br>`reading` | `social`<br>`news`<br>`productivity` | `navigation`<br>`shopping` |
| `HOME_AFTERNOON` | 居家午后 | `O_SHOW_TODAY_TODO`<br>`O_SHOW_SCHEDULE`<br>`R_HOME_RELAX_AND_RESET` | `O_TURN_ON_EYE_COMFORT_MODE`<br>`R_SLEEP_SOON_AND_CHARGE`<br>`O_SHOW_WEATHER` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` | `O_START_CONTEXT_TIMER` | `reading`<br>`entertainment`<br>`music` | `social`<br>`news`<br>`productivity` | `navigation`<br>`shopping` |
| `MORNING_COFFEE` | 早晨咖啡 | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_PLAN_DAY_OVER_COFFEE` | `O_SHOW_WEATHER`<br>`O_SHOW_NEARBY_OPTIONS`<br>`R_ENJOY_LEISURE_MOMENT` | `O_SHOW_BOOKING_DETAILS`<br>`O_TURN_ON_DND` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` | `productivity`<br>`news`<br>`music` | `reading`<br>`entertainment`<br>`social` | `health`<br>`game` |
| `GYM_WORKOUT` | 健身房锻炼 | `O_START_ACTIVITY_TRACKING`<br>`O_START_CONTEXT_TIMER`<br>`R_POST_WORKOUT_RECOVERY` | `R_WORKOUT_PREP`<br>`O_SHOW_WEATHER`<br>`R_OUTDOOR_HYDRATE_AND_REST` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` |  | `health`<br>`music`<br>`productivity` | `news`<br>`reading`<br>`entertainment` | `shopping`<br>`social` |
| `GYM_REST` | 健身房休息 | `O_START_CONTEXT_TIMER`<br>`R_WORKOUT_PREP`<br>`O_START_ACTIVITY_TRACKING` | `O_SHOW_WEATHER`<br>`R_POST_WORKOUT_RECOVERY`<br>`R_OUTDOOR_HYDRATE_AND_REST` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` |  | `health`<br>`music`<br>`productivity` | `news`<br>`reading`<br>`entertainment` | `shopping`<br>`social` |
| `GYM_LONG_STAY` | 健身房长时间停留 | `R_POST_WORKOUT_RECOVERY`<br>`O_START_CONTEXT_TIMER`<br>`O_START_ACTIVITY_TRACKING` | `O_SHOW_WEATHER`<br>`R_WORKOUT_PREP`<br>`R_OUTDOOR_HYDRATE_AND_REST` | `O_SHOW_PAYMENT_QR`<br>`O_SHOW_BOOKING_DETAILS` |  | `health`<br>`music`<br>`productivity` | `news`<br>`reading`<br>`entertainment` | `shopping`<br>`social` |
| `OUTDOOR_RUNNING` | 户外跑步 | `O_START_ACTIVITY_TRACKING`<br>`O_START_CONTEXT_TIMER`<br>`R_OUTDOOR_HYDRATE_AND_REST` | `O_SHOW_WEATHER`<br>`R_WORKOUT_PREP`<br>`R_POST_WORKOUT_RECOVERY` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` |  | `health`<br>`music`<br>`navigation` | `news`<br>`reading`<br>`entertainment` | `shopping`<br>`social` |
| `WEEKEND_OUTDOOR_WALK` | 周末户外步行 | `O_SHOW_WALKING_ROUTE`<br>`O_SHOW_WEATHER`<br>`R_ENJOY_LEISURE_MOMENT` | `R_OUTDOOR_HYDRATE_AND_REST`<br>`O_START_CONTEXT_TIMER`<br>`R_WORKOUT_PREP` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_NEARBY_OPTIONS` | `navigation`<br>`music`<br>`health` | `news`<br>`reading`<br>`entertainment` | `shopping`<br>`social` |
| `CYCLING` | 骑行 | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_WEATHER`<br>`R_COMMUTE_STEADY_AND_SAFE` | `O_START_CONTEXT_TIMER`<br>`R_WORKOUT_PREP`<br>`R_POST_WORKOUT_RECOVERY` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` |  | `health`<br>`navigation`<br>`music` | `news`<br>`reading`<br>`entertainment` | `shopping`<br>`social` |
| `COMMUTE_MORNING` | 早晨通勤到达公司 | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_WORK_START_SETTLE_IN` | `O_SHOW_COMMUTE_TRAFFIC`<br>`R_DEEP_WORK_WINDOW`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` | `R_HOME_RELAX_AND_RESET`<br>`O_SHOW_NEARBY_OPTIONS` | `navigation`<br>`music`<br>`news` | `reading`<br>`social`<br>`health` | `shopping`<br>`game` |
| `COMMUTE_EVENING` | 下班到家 | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_HOME_RELAX_AND_RESET` | `R_WRAP_UP_AND_REST_AFTER_WORK`<br>`O_SHOW_COMMUTE_TRAFFIC`<br>`O_SHOW_WEATHER` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` | `O_START_CONTEXT_TIMER` | `music`<br>`social`<br>`entertainment` | `shopping`<br>`productivity`<br>`reading` | `game`<br>`health` |
| `HOME_TO_GYM` | 从家到达健身房 | `O_START_ACTIVITY_TRACKING`<br>`O_START_CONTEXT_TIMER`<br>`R_WORKOUT_PREP` | `O_SHOW_WEATHER`<br>`R_POST_WORKOUT_RECOVERY`<br>`R_OUTDOOR_HYDRATE_AND_REST` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` |  | `health`<br>`music`<br>`navigation` | `news`<br>`reading`<br>`entertainment` | `shopping`<br>`social` |
| `WAKE_UP` | 家中起床 | `O_SHOW_SCHEDULE`<br>`O_SHOW_WEATHER`<br>`R_WEEKEND_OR_MORNING_GREETING` | `O_SHOW_TODAY_TODO`<br>`R_HOME_RELAX_AND_RESET`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `O_SHOW_BOOKING_DETAILS`<br>`O_START_ACTIVITY_TRACKING` |  | `news`<br>`productivity`<br>`music` | `social`<br>`health`<br>`entertainment` | `navigation`<br>`shopping` |
| `PARK_WALK` | 公园短暂步行 | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_WALKING_ROUTE`<br>`R_ENJOY_LEISURE_MOMENT` | `O_SHOW_WEATHER`<br>`O_START_CONTEXT_TIMER`<br>`R_WORKOUT_PREP` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` |  | `navigation`<br>`health`<br>`music` | `news`<br>`reading`<br>`entertainment` | `shopping`<br>`social` |
| `LONG_DRIVE` | 长距离驾驶 | `R_LONG_DRIVE_REST`<br>`O_SHOW_NEARBY_OPTIONS`<br>`O_SHOW_WEATHER` | `O_SHOW_COMMUTE_TRAFFIC`<br>`O_SHOW_SCHEDULE`<br>`O_SHOW_TRAVEL_INFO` | `O_OPEN_NOTES_QUICK_ENTRY`<br>`O_START_ACTIVITY_TRACKING` |  | `navigation`<br>`music`<br>`news` | `social`<br>`shopping`<br>`productivity` | `game`<br>`health` |
| `CAFE_STAY` | 咖啡厅逗留 | `O_SHOW_PAYMENT_QR`<br>`O_SHOW_SCHEDULE`<br>`R_ENJOY_LEISURE_MOMENT` | `R_DEEP_WORK_WINDOW`<br>`O_SHOW_NEARBY_OPTIONS`<br>`R_MEAL_BREAK` | `O_START_ACTIVITY_TRACKING`<br>`O_TURN_ON_DND` |  | `reading`<br>`productivity`<br>`social` | `music`<br>`news`<br>`entertainment` | `health`<br>`game` |
| `QUIET_HOME` | 居家安静时段 | `R_HOME_RELAX_AND_RESET`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`O_SHOW_SCHEDULE` | `O_SHOW_TODAY_TODO`<br>`R_SLEEP_SOON_AND_CHARGE`<br>`O_SHOW_WEATHER` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` | `O_SHOW_NEARBY_OPTIONS` | `reading`<br>`music`<br>`game` | `social`<br>`news`<br>`productivity` | `navigation`<br>`shopping` |
| `NOISY_GATHERING` | 嘈杂环境聚集 | `O_ENABLE_VIBRATION`<br>`O_INCREASE_NOTIFICATION_VOLUME`<br>`R_NOISY_ENV_STAY_SAFE` | `O_SHOW_PAYMENT_QR`<br>`O_TURN_ON_DND`<br>`R_HOME_RELAX_AND_RESET` | `R_DEEP_WORK_WINDOW`<br>`O_START_ACTIVITY_TRACKING` |  | `social`<br>`music`<br>`navigation` | `shopping`<br>`entertainment`<br>`news` | `reading`<br>`productivity` |
| `LONG_OUTDOOR_WALK` | 长时间户外步行 | `R_OUTDOOR_HYDRATE_AND_REST`<br>`O_SHOW_NEARBY_OPTIONS`<br>`O_START_ACTIVITY_TRACKING` | `O_START_CONTEXT_TIMER`<br>`O_SHOW_WEATHER`<br>`R_WORKOUT_PREP` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` | `R_ENJOY_LEISURE_MOMENT` | `health`<br>`navigation`<br>`music` | `news`<br>`reading`<br>`entertainment` | `shopping`<br>`social` |
| `LATE_NIGHT_NOISY` | 深夜嘈杂场所 | `O_ENABLE_VIBRATION`<br>`O_INCREASE_NOTIFICATION_VOLUME`<br>`R_NOISY_ENV_STAY_SAFE` | `O_TURN_ON_DND`<br>`R_HOME_RELAX_AND_RESET`<br>`R_AFTER_MEAL_WALK` | `O_START_ACTIVITY_TRACKING`<br>`R_DEEP_WORK_WINDOW` |  | `navigation`<br>`social`<br>`music` | `shopping`<br>`entertainment`<br>`news` | `reading`<br>`productivity` |
| `DELIVERY_AT_OFFICE` | 公司收到快递 | `O_SHOW_PICKUP_DETAILS`<br>`R_DELIVERY_PICKUP_REMINDER`<br>`O_SHOW_TODAY_TODO` | `R_DEEP_WORK_WINDOW`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`R_WORK_START_SETTLE_IN` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` |  | `shopping`<br>`productivity`<br>`social` | `reading`<br>`health`<br>`navigation` | `game`<br>`entertainment` |
| `EDUCATION_WAITING` | 辅导班等待 | `O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO`<br>`R_WAITING_TIME_USEFUL` | `R_DEEP_WORK_WINDOW`<br>`O_TURN_ON_SILENT_MODE`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` | `R_HOME_RELAX_AND_RESET` | `reading`<br>`productivity`<br>`game` | `news`<br>`health`<br>`social` | `shopping`<br>`navigation` |
| `EDUCATION_LONG_SIT` | 辅导班久坐 | `R_WORK_BREAK_AND_STRETCH`<br>`O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO` | `R_DEEP_WORK_WINDOW`<br>`O_TURN_ON_SILENT_MODE`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` |  | `reading`<br>`productivity`<br>`health` | `news`<br>`social`<br>`entertainment` | `shopping`<br>`navigation` |
| `HOME_LUNCH` | 居家午餐 | `R_MEAL_BREAK`<br>`O_SHOW_NEARBY_OPTIONS`<br>`O_SHOW_PAYMENT_QR` | `R_ENJOY_LEISURE_MOMENT`<br>`O_SHOW_WEATHER`<br>`O_SHOW_BOOKING_DETAILS` | `O_TURN_ON_DND`<br>`O_OPEN_NOTES_QUICK_ENTRY` | `O_START_ACTIVITY_TRACKING` | `shopping`<br>`social`<br>`entertainment` | `music`<br>`news`<br>`reading` | `health`<br>`productivity` |
| `AFTER_LUNCH_WALK` | 饭后散步 | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_WALKING_ROUTE`<br>`R_AFTER_MEAL_WALK` | `O_SHOW_WEATHER`<br>`O_START_CONTEXT_TIMER`<br>`R_WORKOUT_PREP` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` | `R_HOME_RELAX_AND_RESET`<br>`O_SHOW_NEARBY_OPTIONS` | `health`<br>`navigation`<br>`music` | `news`<br>`reading`<br>`entertainment` | `shopping`<br>`social` |
| `MEETING_UPCOMING` | 会议即将开始 | `O_SHOW_MEETING_DETAILS`<br>`O_TURN_ON_DND`<br>`R_PREPARE_FOR_MEETING` | `O_SHOW_SCHEDULE`<br>`O_OPEN_NOTES_QUICK_ENTRY`<br>`R_CAPTURE_MEETING_FOLLOW_UPS` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` | `O_SHOW_NEARBY_OPTIONS`<br>`O_SHOW_TRAVEL_INFO` | `productivity`<br>`navigation`<br>`social` | `reading`<br>`music`<br>`news` | `shopping`<br>`game` |
| `IN_MEETING` | 会议进行中 | `O_TURN_ON_DND`<br>`O_OPEN_NOTES_QUICK_ENTRY`<br>`R_CAPTURE_MEETING_FOLLOW_UPS` | `O_SHOW_MEETING_DETAILS`<br>`R_PREPARE_FOR_MEETING`<br>`O_SHOW_SCHEDULE` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` |  | `productivity`<br>`social`<br>`reading` | `music`<br>`news`<br>`health` | `shopping`<br>`game` |
| `REMOTE_MEETING` | 异地会议出发 | `O_SHOW_TRAVEL_INFO`<br>`O_SHOW_MEETING_DETAILS`<br>`R_DEPART_EARLY_FOR_OFFSITE` | `O_SHOW_SCHEDULE`<br>`O_SHOW_BOOKING_DETAILS`<br>`R_CHECK_TRIP_OR_BOOKING_INFO` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` | `O_TURN_ON_DND`<br>`R_HOME_RELAX_AND_RESET` | `navigation`<br>`productivity`<br>`social` | `shopping`<br>`music`<br>`reading` | `game`<br>`health` |
| `NO_MEETINGS` | 今日无会议 | `O_SHOW_TODAY_TODO`<br>`R_DEEP_WORK_WINDOW`<br>`O_SHOW_SCHEDULE` | `O_TURN_ON_EYE_COMFORT_MODE`<br>`R_WORK_START_SETTLE_IN`<br>`R_WORK_BREAK_AND_STRETCH` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` | `O_SHOW_NEARBY_OPTIONS` | `productivity`<br>`music`<br>`reading` | `social`<br>`health`<br>`navigation` | `shopping`<br>`game` |
| `CAL_HAS_EVENTS` | 今日日程提醒 | `O_SHOW_SCHEDULE`<br>`R_CALENDAR_QUICK_LOOK`<br>`O_SHOW_TODAY_TODO` | `R_DEEP_WORK_WINDOW`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`R_WORK_START_SETTLE_IN` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` | `O_SHOW_TRAVEL_INFO`<br>`O_TURN_ON_DND`<br>`R_HOME_RELAX_AND_RESET` | `productivity`<br>`navigation`<br>`news` | `reading`<br>`social`<br>`health` | `shopping`<br>`game` |
| `RETURN_OFFICE_AFTER_LUNCH` | 午餐后回到办公室 | `O_SHOW_TODAY_TODO`<br>`O_SHOW_SCHEDULE`<br>`R_WORK_START_SETTLE_IN` | `R_WORK_BREAK_AND_STRETCH`<br>`R_DEEP_WORK_WINDOW`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` | `O_SHOW_NEARBY_OPTIONS` | `productivity`<br>`music`<br>`news` | `reading`<br>`social`<br>`health` | `shopping`<br>`game` |
| `HOME_AFTER_GYM` | 健身后回家 | `R_POST_WORKOUT_RECOVERY`<br>`O_START_CONTEXT_TIMER`<br>`R_HOME_RELAX_AND_RESET` | `O_SHOW_WEATHER`<br>`R_WORKOUT_PREP`<br>`O_START_ACTIVITY_TRACKING` | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR` |  | `health`<br>`music`<br>`entertainment` | `news`<br>`reading`<br>`productivity` | `shopping`<br>`social` |
| `COMMUTE_FROM_HOME` | 从家出发通勤中 | `O_SHOW_COMMUTE_TRAFFIC`<br>`R_COMMUTE_STEADY_AND_SAFE`<br>`O_SHOW_SCHEDULE` | `O_SHOW_WEATHER`<br>`O_SHOW_TRAVEL_INFO`<br>`R_CHECK_TRIP_OR_BOOKING_INFO` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` | `O_SHOW_BOOKING_DETAILS` | `navigation`<br>`music`<br>`news` | `social`<br>`shopping`<br>`productivity` | `game`<br>`health` |
| `OFFICE_TO_CAFE` | 从办公室到咖啡馆 | `O_SHOW_PAYMENT_QR`<br>`R_ENJOY_LEISURE_MOMENT`<br>`O_SHOW_NEARBY_OPTIONS` | `R_MEAL_BREAK`<br>`O_SHOW_WEATHER`<br>`O_SHOW_BOOKING_DETAILS` | `O_TURN_ON_DND`<br>`O_OPEN_NOTES_QUICK_ENTRY` |  | `social`<br>`shopping`<br>`reading` | `music`<br>`news`<br>`entertainment` | `health`<br>`productivity` |
| `LATE_NIGHT_PHONE` | 深夜躺姿过长 | `R_SLEEP_SOON_AND_CHARGE`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`O_TURN_ON_DND` | `R_HOME_RELAX_AND_RESET`<br>`O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` |  | `health`<br>`reading`<br>`music` | `social`<br>`news`<br>`productivity` | `navigation`<br>`shopping` |
| `OFFICE_FOCUS_LONG` | 办公手机扣放久坐 | `R_WORK_BREAK_AND_STRETCH`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`R_DEEP_WORK_WINDOW` | `R_WORK_START_SETTLE_IN`<br>`O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` |  | `productivity`<br>`music`<br>`health` | `reading`<br>`social`<br>`navigation` | `shopping`<br>`game` |
| `HOME_DARK_LONG` | 居家白天暗环境过长 | `R_OPEN_CURTAINS_OR_LIGHT`<br>`O_SHOW_WEATHER`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `R_HOME_RELAX_AND_RESET`<br>`O_SHOW_SCHEDULE`<br>`O_SHOW_TODAY_TODO` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` |  | `health`<br>`reading`<br>`music` | `social`<br>`news`<br>`productivity` | `navigation`<br>`shopping` |
| `UNKNOWN_LONG_STAY` | 未知地点久留 | `O_PROMPT_ADD_FREQUENT_PLACE`<br>`O_SHOW_NEARBY_OPTIONS`<br>`O_SHOW_WEATHER` | `O_SHOW_SCHEDULE`<br>`R_AFTER_MEAL_WALK`<br>`R_CALENDAR_QUICK_LOOK` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` |  | `navigation`<br>`social`<br>`shopping` | `music`<br>`reading`<br>`news` | `game`<br>`productivity` |
| `OFFICE_LONG_SESSION` | 办公超长连续 | `R_WORK_BREAK_AND_STRETCH`<br>`O_SHOW_TODAY_TODO`<br>`O_TURN_ON_EYE_COMFORT_MODE` | `R_DEEP_WORK_WINDOW`<br>`R_WORK_START_SETTLE_IN`<br>`O_SHOW_SCHEDULE` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` | `O_SHOW_NEARBY_OPTIONS` | `productivity`<br>`health`<br>`music` | `reading`<br>`social`<br>`navigation` | `shopping`<br>`game` |
| `HOME_EVENING_DARK` | 居家傍晚暗环境 | `O_TURN_ON_EYE_COMFORT_MODE`<br>`R_HOME_RELAX_AND_RESET`<br>`O_SHOW_SCHEDULE` | `O_SHOW_TODAY_TODO`<br>`R_SLEEP_SOON_AND_CHARGE`<br>`O_SHOW_WEATHER` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` | `O_START_CONTEXT_TIMER` | `reading`<br>`entertainment`<br>`music` | `social`<br>`news`<br>`productivity` | `navigation`<br>`shopping` |
| `HOME_EVENING_NOISY` | 居家傍晚嘈杂 | `O_INCREASE_NOTIFICATION_VOLUME`<br>`O_ENABLE_VIBRATION`<br>`R_HOME_RELAX_AND_RESET` | `R_NOISY_ENV_STAY_SAFE`<br>`O_TURN_ON_DND`<br>`R_AFTER_MEAL_WALK` | `O_START_ACTIVITY_TRACKING`<br>`R_DEEP_WORK_WINDOW` |  | `social`<br>`music`<br>`entertainment` | `shopping`<br>`news`<br>`health` | `reading`<br>`productivity` |
| `SCHOOL_QUIET` | 学校安静时段 | `O_TURN_ON_SILENT_MODE`<br>`R_DEEP_WORK_WINDOW`<br>`O_SHOW_SCHEDULE` | `O_SHOW_TODAY_TODO`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`R_AFTER_MEAL_WALK` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` |  | `reading`<br>`productivity`<br>`music` | `news`<br>`health`<br>`social` | `shopping`<br>`navigation` |
| `CAFE_QUIET` | 咖啡馆安静 | `O_TURN_ON_SILENT_MODE`<br>`O_SHOW_SCHEDULE`<br>`R_DEEP_WORK_WINDOW` | `O_SHOW_TODAY_TODO`<br>`O_TURN_ON_EYE_COMFORT_MODE`<br>`R_AFTER_MEAL_WALK` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` | `R_ENJOY_LEISURE_MOMENT`<br>`O_SHOW_NEARBY_OPTIONS` | `reading`<br>`productivity`<br>`music` | `news`<br>`health`<br>`social` | `shopping`<br>`navigation` |
| `TRAIN_DEPARTURE` | 高铁出行提醒 | `O_SHOW_TRAVEL_INFO`<br>`R_CHECK_TRIP_OR_BOOKING_INFO`<br>`O_SHOW_BOOKING_DETAILS` | `O_SHOW_WEATHER`<br>`O_SHOW_SCHEDULE`<br>`R_AFTER_MEAL_WALK` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` |  | `navigation`<br>`news`<br>`productivity` | `social`<br>`shopping`<br>`music` | `game`<br>`health` |
| `FLIGHT_BOARDING` | 航班登机提醒 | `O_SHOW_TRAVEL_INFO`<br>`R_CHECK_TRIP_OR_BOOKING_INFO`<br>`O_SHOW_BOOKING_DETAILS` | `O_SHOW_WEATHER`<br>`O_SHOW_SCHEDULE`<br>`R_AFTER_MEAL_WALK` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` |  | `navigation`<br>`news`<br>`productivity` | `social`<br>`shopping`<br>`music` | `game`<br>`health` |
| `HOTEL_CHECKIN` | 酒店入住提醒 | `O_SHOW_BOOKING_DETAILS`<br>`R_CHECK_TRIP_OR_BOOKING_INFO`<br>`O_SHOW_TRAVEL_INFO` | `O_SHOW_WEATHER`<br>`O_SHOW_SCHEDULE`<br>`R_AFTER_MEAL_WALK` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` |  | `navigation`<br>`social`<br>`productivity` | `shopping`<br>`music`<br>`reading` | `game`<br>`health` |
| `MOVIE_TICKET` | 电影取票提醒 | `O_SHOW_BOOKING_DETAILS`<br>`O_SHOW_PAYMENT_QR`<br>`R_CHECK_TRIP_OR_BOOKING_INFO` | `O_SHOW_NEARBY_OPTIONS`<br>`R_ENJOY_LEISURE_MOMENT`<br>`R_MEAL_BREAK` | `O_START_ACTIVITY_TRACKING`<br>`O_TURN_ON_DND` |  | `entertainment`<br>`social`<br>`shopping` | `music`<br>`news`<br>`reading` | `health`<br>`productivity` |
| `HOSPITAL_APPOINTMENT` | 就诊提醒 | `O_SHOW_BOOKING_DETAILS`<br>`R_CHECK_TRIP_OR_BOOKING_INFO`<br>`O_SHOW_TRAVEL_INFO` | `O_SHOW_WEATHER`<br>`O_SHOW_SCHEDULE`<br>`R_AFTER_MEAL_WALK` | `O_SHOW_PAYMENT_QR`<br>`O_START_ACTIVITY_TRACKING` |  | `health`<br>`navigation`<br>`productivity` | `social`<br>`shopping`<br>`music` | `game`<br>`entertainment` |
| `RIDESHARE_PICKUP` | 网约车接驾提醒 | `O_SHOW_TRAVEL_INFO`<br>`R_CHECK_TRIP_OR_BOOKING_INFO`<br>`O_SHOW_BOOKING_DETAILS` | `O_SHOW_WEATHER`<br>`O_SHOW_SCHEDULE`<br>`R_AFTER_MEAL_WALK` | `O_START_ACTIVITY_TRACKING`<br>`O_SHOW_PAYMENT_QR` |  | `navigation`<br>`social`<br>`music` | `shopping`<br>`reading`<br>`productivity` | `game`<br>`health` |

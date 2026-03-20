# App Category Agent 设计说明

## 1. 目标

这个 agent 的目标是做 **app category 排序**，而不是像现有 R/O agent 那样推荐一个单独动作。

给定当前 state/context，这个 agent 应该：
- 对 10 个 app category 打分
- 按相关性从高到低排序
- 向用户展示 top 6
- 根据展示顺序和点击反馈持续学习

排在越前面的 app category，表示系统认为它越值得推荐。

## 2. 为什么建议单独一个 agent

我建议把它做成一个和现有 R/O bandit 分开的 agent，即使两者可以共享同一套 state/context feature。

原因是：
- R/O agent 推荐的是**一个动作**
- app agent 推荐的是**一个有序列表**
- 反馈形式不同，这里要关心的是**展示顺序、点击了哪个、点击位置、以及没点击**
- 优化目标也不同，这里更接近 **ranking / slate recommendation**，而不是单个 action 的推荐

所以两者可以共享：
- context/state feature
- 用户历史特征
- logging pipeline
- 训练基础设施

但最好分开设计：
- action space
- reward
- serving 逻辑
- evaluation 指标

## 3. 推荐的整体路线

### V0-App
先做一个 **contextual scorer + top-k ranking + 小探索**。

这是第一版最推荐的做法，因为实现简单、容易控制，也很适合 demo / beta 测试。

### V1-App
加入 **基于用户历史特征的个性化**。

### V2-App
如果后面确实需要，再考虑更复杂的排序 bandit，例如：
- cascading bandit
- position-based bandit
- slate bandit

但第一版不建议直接上这些。

## 4. 为什么第一版不建议直接做 cascading bandit

cascading bandit 和这个问题确实相关，因为用户通常会从上往下浏览列表，看到合适的项后可能就点击并停止。

但它不适合作为第一版的原因是：
- 会引入更多假设，比如用户一定按顺序看列表
- 后面的 item 可能是“没点”还是“根本没看到”，解释会更复杂
- 曝光和反馈的定义需要更小心
- 实现和调试成本更高

所以第一版更推荐先做一个简单、可控的 ranking 系统，把数据收起来。

## 5. 推荐的 V0 模型

### 核心思路
给每个 app category 维护一个 scorer。

每次请求时：
1. 构造 context vector
2. 对 10 个 app category 分别打分
3. 按分数排序
4. 展示 top 6
5. 记录曝光顺序和点击反馈

### 推荐算法
建议先用 **scenario-aware top-k LinUCB ranking**。

也就是：
- 输入还是共享的 state/context feature
- 每个 app category 对应一个 LinUCB 风格的 scorer
- 对每个 category 计算一个分数
- 按分数排序
- 展示 top 6

如果你们现有 R/O agent 也是 LinUCB，这会是最自然、最统一的方案。

## 6. default app prior

你说每个 scenario 已经选了一个“直觉上最合适的 app category”，这个非常有价值，可以作为冷启动时的 **default prior**。

冷启动时可以把最终分数写成：

`final_score(category) = model_score(category) + default_bonus(如果这个 category 是该 scenario 的默认 app)`

这样做有两个好处：
- 冷启动时列表从一个合理的场景先验开始
- 其他 category 仍然有机会被探索，并在数据足够时超过 default

这和前面 R/O agent 的 V0 思路是一致的。

## 7. exploration 策略

app list 不应该完全随机，但也不建议完全贪心。

推荐做法是：
- 前 1 到 2 个位置尽量稳定
- 第 3 到 6 个位置留一些探索空间

可以有两种实现：

### 方案 A：按 UCB 分数排序
对每个 category 算 UCB score，然后直接按 UCB 分数排序。

### 方案 B：分数排序 + 控制式扰动
- 前 1 到 2 个高分 category 固定
- 剩余位置从 next-best categories 中做一点随机采样

在 demo 场景里，这种方式通常更容易控制体验。

## 8. 反馈和 reward 设计

每次展示都应该记录：
- 展示的 top 6 category
- 展示顺序
- 用户点击了哪个 category（如果有）
- 点击位置（如果有）
- 是否 no-click

### 第一版推荐的 reward 理解方式
如果用户点击了第 `j` 个位置的 category：
- 被点击的 category：reward = 1
- 排在它前面的 item：reward = 0 或小负值
- 排在它后面的 item：先当作未观察，不要急着记强负

如果用户一个都没点：
- 先把展示的 item 记为 0
- 第一版不要急着当作强负样本

原因是：
- no-click 不一定表示排序差
- 后面的 item 可能根本没被看到
- 过早引入强负反馈容易把模型带偏

## 9. 日志要求

每次 serving event，建议记录：
- user_id
- timestamp
- scenario_id
- context feature vector
- 如果方便，记录 10 个 app category 的原始分数
- 展示的 top 6 category
- 展示位置
- clicked category
- clicked position
- no-click flag
- model version
- 如果方便，也记录 exploration metadata / propensity

这样后面无论你想继续做简单 ranker，还是升级到更复杂的 ranking bandit，数据都够用。

## 10. feature 设计

app agent 可以复用和 R/O agent 相同的 state/context backbone。

推荐的 feature 分组：

### A. 当前状态 / context
- scenario
- time / location / motion / phone / light / sound / day-type
- 当前 state code
- 上一个 state code
- duration 特征
- battery / charging / mobility / calendar 等即时上下文

### B. app 相关的用户历史特征
建议先加这些：
- overall app-click rate
- recent click rate
- recent no-click count
- 当前 scenario 下的 click rate
- 当前 scenario 最近点击的 app category
- 当前 scenario 下各 app category 的点击次数
- 全局各 app category 的点击次数

这一步已经能实现 feature-based personalization，而不用一开始就做 per-user model。

## 11. 个性化路线

### V1-App：基于 feature 的个性化
先保持一个全局模型，但在 context 里加入用户历史特征。

这样虽然模型参数还是共享的，但不同用户因为历史特征不同，会得到不同的排序结果。

### V2-App：per-user residual personalization
如果后面真实数据足够多，再在全局模型上叠加一个用户级 residual。

概念上可以写成：

`score(user, context, category) = global_score(context, category) + user_residual(user, context, category)`

推荐复杂度顺序：
1. 只有全局模型
2. 全局模型 + 用户历史特征
3. 全局模型 + 用户 residual

不建议一开始就直接跳到完全独立的 per-user model。

## 12. 评估指标

第一版建议先看这些：
- displayed top-6 的 click-through rate
- top-1 click rate
- mean clicked position
- no-click impressions 的比例
- scenario-level click rate
- 相比 default-only 排序 baseline 的提升

如果后面需要，再加 ranking 指标，例如 MRR 或 NDCG。

## 13. 实际实现建议

第一版要做一个能工作的系统，我建议：
- 单独做一个 App Category Agent
- 复用同样的 state/context pipeline
- 每个 app category 一个 scorer
- 对 10 个 category 全部打分后展示 top 6
- 加一个 scenario-default app prior
- 加一点小探索
- 完整记录 impression 和 click 数据
- 先从全局模型开始
- 后续再加入用户历史特征做个性化

## 14. 最终建议

最实际的路线是：

### V0-App
**Scenario-aware top-k LinUCB ranking + default app prior + 小探索**

### V1-App
**全局 ranking model + 用户历史特征**

### V2-App
**全局模型 + per-user residual**，但应在真实数据足够以后再做

这样你可以从一个稳定的冷启动系统，逐步演进到个性化 app 排序，而不会在第一版就引入过多复杂度。

# 面向 Scenario 的 R/O 推荐 Bandit 设计说明

这份说明是一个简短的实现路线建议，围绕下面三步展开：

- **V0**：masking + default prior + 小探索
- **V1-A**：全局 LinUCB + 用户历史统计特征
- **V1-B**：全局 LinUCB + 用户个性化残差

这份文档故意不写得太细。目标是给出清晰可落地的方向，而不是替代最终实现设计。

---

## 1. 整体思路

当前系统本身已经有很强的产品结构：

- 一共有 **65 个 scenario**
- 一共有 **167 个 R/O action**
- 每个 scenario 都有：
  - 一个合法动作集合：`actionIDs`
  - 一个默认动作：`actionIDDefault`
- 目前 **A / APP action 不进入这个 bandit**

所以，这个问题应该被看作一个 **带 masking 的 contextual bandit**，而不是一个平铺的 167 动作 bandit。

也就是说：

1. 先确定当前 scenario
2. 再取出这个 scenario 对应的合法动作集合
3. 只在这个合法集合里打分和选择
4. 非法动作根本不参与排序

换句话说，这个 bandit 真正回答的问题是：

> 在当前 scenario 和当前 context 下，现在应该推荐哪个合法的 R/O action？

---

## 2. 开始实现前的通用建议

下面几点不管是 V0 还是 V1 都很重要。

### reward 定义
reward 定义最好尽早固定。

当前简单版本可以是：

- `thumbs_up = 1.0`
- `thumbs_down = -0.5`
- `no_feedback = 0`

这已经是一个不错的起点。

后面如果要细化，也可以再考虑加入下游行为，例如：

- 接受后是否真的执行了操作
- 是否点开后忽略
- 是否立即拒绝

### 日志记录
从第一天开始，建议每一次推荐都记录下面这些信息：

- `scenario_id`
- context 特征
- 当前合法动作集合
- 最终选择的动作
- 如果可以，记录每个合法动作的分数
- 被展示动作的 exploration probability / propensity
- 用户反馈
- 最终 reward

这样后续做离线评估、重训、off-policy 分析都会轻松很多。

### feature 范围
第一版不要一上来塞太多特征。

比较实用的起步方式是：

- 当前 scenario / state 特征
- 少量即时 context 特征
- 后续再加少量用户历史统计特征

---

# V0：Masking + Default Prior + 小探索

## 3. V0 的目标

V0 是一个**安全冷启动策略**。

它的目标不是完全个性化，而是：

- 推荐在当前 scenario 下合理的动作
- 大多数时候优先推荐默认动作
- 但也保留一点探索，收集其他动作的真实反馈
- 为后续 V1 积累可训练的 bandit 日志

## 4. action masking 怎么做

### 在部署时
在线 serving 时，流程应该是：

1. 获取当前 scenario `s`
2. 查出该 scenario 的合法动作集合 `A(s)`
3. 只对 `A(s)` 里的动作打分
4. 从这个集合中选出一个动作

用纯文本表示就是：

`chosen action = argmax over a in A(s) of [ predicted_reward(x, a) + exploration_bonus(x, a) ]`

其中：

- `x` 是当前 context 向量
- `A(s)` 是当前 scenario 的合法动作集合

要注意：

- 非法动作不是“打低分”
- 而是“根本不进入候选集”

### 在训练时
训练时也要保持同样的逻辑。

每条样本应该包含：

- 当前 `scenario_id`
- 该 scenario 下被展示的合法动作集合
- 被选中的动作
- 该动作的 reward

不要把训练做成“所有样本都默认可以从 167 个动作里选”。

## 5. V0 的 synthetic data 怎么构造

synthetic data 最好尽量长得像未来的真实线上日志。

建议的 synthetic sample 字段：

- `scenario_id`
- `state_features`
- 如果有的话，`state_sequence`
- `shown_actions` = 该 scenario 的全部合法动作
- `selected_action`
- `reward`

这比只做一个“scenario -> default label”的简单表更好，因为线上真实系统本来就有候选动作集合和 reward 日志。

## 6. synthetic data 里 default 和 non-default 怎么分配

不要把 synthetic data 做成 100% default-only。

建议比例：

- **70% 到 85%** 的 synthetic sample：选择 default action
- **15% 到 30%** 的 synthetic sample：选择同一 scenario 下的其他合法动作

原因是：

- 以 default 为主，可以保留产品先验
- 留一部分 non-default，可以避免模型退化成纯规则表

更好的做法是：

- 如果可以，non-default 不要纯随机
- 而是用一些简单的 context 条件规则去选

例如：

- 在 `ARRIVE_OFFICE` 下，默认可能是 `show_schedule`
- 但在某些 context 下，`show_todo_list` 也可以被视为正样本

## 7. V0 在线推荐策略怎么做

一个比较好的 V0 策略是：

- 先做 scenario masking
- 强烈偏向 default action
- 加少量探索

这里有两个可行实现。

### 方案 A：简单的 default-biased policy
先不引入 learned scorer，直接按概率做：

- default action：80% 到 90%
- 其他合法动作：共享剩下的 10% 到 20%

这非常简单，而且已经可以作为不错的冷启动 baseline。

### 方案 B：带 default prior 的 masked LinUCB
在合法动作集合内用 LinUCB 打分，但给 default action 一个额外加分。

用纯文本公式表示：

`score(a) = predicted_reward(x, a) + alpha * uncertainty_bonus(x, a) + beta * is_default(a)`

其中：

- `alpha` 控制探索强度
- `beta` 是给默认动作的正向偏置
- `is_default(a)` 在动作 `a` 是当前 scenario 默认动作时取 1，否则取 0

如果团队已经在实现 LinUCB，这通常是更好的 V0 形态。

---

# V1-A：全局 LinUCB + 用户历史统计特征

## 8. 目标

V1-A 的目标是在**不引入 per-user 单独参数**的前提下，先实现一版 feature-based personalization。

也就是说：

- 仍然只有 **一个共享的全局模型**
- 个性化来自于 context 中加入了用户历史统计特征

因此，不同用户虽然共用同一套模型参数，但由于输入不同，最终推荐结果也会不同。

## 9. 什么叫 feature-based personalization

核心思想就是：

- 模型还是同一个
- 但不同用户有不同的历史行为特征
- 所以模型会输出不同的动作分数

例如模型可以利用这些特征：

- 这个用户平时更容易接受 `R` action
- 这个用户在晚上经常忽略 `O` action
- 这个用户在办公室相关 scenario 下经常接受 schedule 类动作

所以即使只有一个全局模型，也已经能做出一定程度的个性化。

## 10. 推荐的用户历史特征分组

### A. 用户整体偏好
描述用户整体行为：

- overall accept rate
- overall deny rate
- recent-N accept rate
- recent consecutive ignore count
- recent consecutive deny count

### B. action type 偏好
因为当前主动作空间就是 R/O，所以这组很有价值：

- accept rate for `R`
- accept rate for `O`
- deny rate for `R`
- deny rate for `O`

### C. 当前 scenario 下的偏好
这组通常最有用：

- accept rate in current scenario
- deny rate in current scenario
- recent feedback score in current scenario
- count of prior exposures in current scenario
- recently accepted action type in current scenario

## 11. V1-A 怎么训练

训练方式仍然比较简单：

- 只有一个全局 LinUCB 模型
- 和 V0 一样继续做 scenario-based masking
- 输入用基础 context 特征 + 用户历史统计特征
- 用真实用户 reward 做在线更新

这是最稳妥的第一步个性化方案。

---

# V1-B：全局 LinUCB + 用户个性化残差

## 12. 目标

V1-B 在 V1-A 的基础上，再进一步做**显式的 per-user personalization**。

思路是：

- 保留一个全局模型，学习所有用户的共性
- 在全局模型之上，再加一个用户自己的小修正

用纯文本写就是：

`final_score(user, x, a) = global_score(x, a) + user_residual_score(user, x, a)`

也可以写成：

`score(user, x, a) = x · theta_global[a] + x · delta_user[a]`

其中：

- `theta_global[a]` 是 action `a` 的全局共享参数
- `delta_user[a]` 是用户自己的修正参数

## 13. 为什么这样比“每个用户单独训一个模型”更好

如果直接给每个用户训一个完整独立模型，通常会非常吃样本。

因为大多数用户都会存在：

- 很多 scenario 曝光很少
- 很多 action 曝光很少
- reward 信号稀疏

所以更好的办法是：

- 全局模型学共性
- 用户残差只学“这个用户和平均用户有什么不同”

## 14. 全局模型 + 用户残差怎么训练

### 全局部分
全局模型用所有用户的数据训练。

它学习的是群体层面的 context 到 reward 的映射。

### 用户残差部分
对于每个用户：

- 残差初始为 0
- 只用该用户自己的交互数据更新
- 要强烈 regularize 到 0
- 只有当用户数据足够多时，才逐步提高它的影响力

比较实用的经验规则：

- 用户数据很少 -> 主要依赖全局模型
- 用户数据中等 -> 全局 + 残差混合
- 用户数据较多 -> 可以更信任残差

## 15. 用户残差的粒度建议

不要一上来就做最细的版本。

建议顺序：

1. user + scenario residual
2. 再考虑 user + action-type residual
3. 只有数据足够多时，才做 user + action residual

如果一开始就做完整的 user × 167 action residual，往往会太稀疏。

---

# 16. 推荐 rollout 路线

## Phase 1
先上线 **V0**：

- scenario masking
- 强 default prior
- 小探索
- 完整记录日志

## Phase 2
训练 **V1-A**：

- 全局 LinUCB
- 加用户历史统计特征
- 仍然只有一个共享模型

## Phase 3
升级到 **V1-B**：

- 保留全局模型
- 加 per-user residual
- 随着用户数据增多，逐步提高 residual 的作用

这条分阶段路线，比直接跳到完全 per-user 模型要稳很多，也更容易实现。

---

# 17. 最终建议

一个比较好的实际路线是：

## V0
**带 strong default prior 和小探索的 masked recommendation**

- 用 scenario 做 valid-action masking
- 让 default action 大多数时候胜出
- 但也偶尔探索其他合法动作
- synthetic training data 以 default 为主，但不要 100% default

## V1-A
**全局 LinUCB + 用户历史统计特征**

- 一个共享模型
- 通过行为历史摘要特征实现个性化

## V1-B
**全局 LinUCB + per-user residual**

- 共享全局 backbone
- 每个用户有自己的小修正
- 用 regularization 和 data-threshold gating 控制稳定性

这条路线的优点是：

- 冷启动安全
- 线上反馈可学习
- 可以逐步过渡到真正的个性化推荐

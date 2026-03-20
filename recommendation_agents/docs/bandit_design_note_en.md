# Bandit Design Note for Scenario-Based R/O Recommendation

This note is a short implementation guide for the bandit roadmap:

- **V0**: masking + default prior + small exploration
- **V1-A**: global LinUCB + user-history features
- **V1-B**: global LinUCB + per-user residual

It is intentionally high-level. The goal is to suggest a practical direction, not to over-specify the final implementation.

---

## 1. Big picture

The system already has a strong product structure:

- There are **65 scenarios**.
- There are **167 R/O actions**.
- Each scenario has:
  - a list of valid actions: `actionIDs`
  - one default action: `actionIDDefault`
- For now, **A / APP actions stay outside** this bandit.

So this should be treated as a **masked contextual bandit**, not as one flat 167-action bandit.

That means:

1. first determine the current scenario
2. load the valid action set for that scenario
3. only score actions inside that valid set
4. never rank invalid actions

In other words, the bandit only answers:

> Given the current scenario and context, which valid R/O action should we recommend now?

---

## 2. General recommendations before implementation

A few things are important regardless of V0 or V1.

### Reward definition
Keep the reward definition stable as early as possible.

Current simple version:

- `thumbs_up = 1.0`
- `thumbs_down = -0.5`
- `no_feedback = 0`

This is a good starting point.

Later, the team can decide whether to refine reward with downstream behavior, such as:

- accepted and actually executed
- opened but ignored
- denied immediately

### Logging
From day 1, log these fields for every recommendation event:

- `scenario_id`
- context features
- valid action set
- chosen action
- score of each valid action if feasible
- exploration probability / propensity of the chosen action
- user feedback
- final reward

This will make later offline evaluation and retraining much easier.

### Feature scope
Do not put too many features into the first version.

A practical starting point is:

- current scenario / state features
- a few immediate context features
- later, a small number of user-history features

---

# V0: Masking + Default Prior + Small Exploration

## 3. V0 goal

V0 is a **safe cold-start policy**.

Its job is not to be fully personalized. Its job is to:

- recommend actions that make sense in the current scenario
- usually prefer the default action
- still explore a little so the system can collect feedback on other actions
- produce useful logged data for V1

## 4. How action masking works

### In serving
At serving time:

1. get the current scenario `s`
2. look up the valid action set `A(s)`
3. compute scores only for actions in `A(s)`
4. choose one action from that set

A plain-text version of the idea is:

`chosen action = argmax over a in A(s) of [ predicted_reward(x, a) + exploration_bonus(x, a) ]`

where:

- `x` is the current context vector
- `A(s)` is the valid action set for the current scenario

Important:

- invalid actions are not given a low score
- invalid actions are simply not in the candidate set

### In training
Training should follow the same logic.

Each sample should contain:

- current `scenario_id`
- valid actions shown for that scenario
- selected action among those valid actions
- reward for that selected action

Do **not** train as if all 167 actions are available for every sample.

## 5. How to build synthetic data for V0

Synthetic data should look similar to future online logs.

Recommended synthetic sample fields:

- `scenario_id`
- `state_features`
- `state_sequence` if available
- `shown_actions` = all valid actions for that scenario
- `selected_action`
- `reward`

This is better than a simple dataset that only says:

- scenario -> default label

because the real online system will also have candidate sets and logged rewards.

## 6. How often synthetic data should use default vs non-default

Do **not** make synthetic data 100% default-only.

Recommended mix:

- **70% to 85%** of synthetic samples: select the default action
- **15% to 30%** of synthetic samples: select another valid action in the same scenario

Why:

- default-heavy data preserves the product prior
- some non-default data prevents the model from becoming a pure rule table

Better version:

- when possible, pick non-default actions using simple context rules
- do not make the non-default portion purely random if you can avoid it

Example:

- in `ARRIVE_OFFICE`, default may be `show_schedule`
- but in some contexts, `show_todo_list` can also be positive

## 7. How V0 should serve recommendations

The simplest good V0 policy is:

- mask to valid actions first
- strongly prefer the default action
- add a small amount of exploration

There are two good implementation choices.

### Option A: simple default-biased policy
Without a learned scorer yet:

- default action: 80% to 90%
- other valid actions: share the remaining 10% to 20%

This is very simple and can already be a good cold-start baseline.

### Option B: masked LinUCB with default prior
Use a LinUCB score among valid actions, but add extra bias to the default action.

Plain-text formula:

`score(a) = predicted_reward(x, a) + alpha * uncertainty_bonus(x, a) + beta * is_default(a)`

where:

- `alpha` controls exploration
- `beta` is a positive bonus for the scenario default action
- `is_default(a)` is 1 if action `a` is the default for this scenario, otherwise 0

This is likely the best V0 shape if LinUCB is already being implemented.

---

# V1-A: Global LinUCB + User-History Features

## 8. Goal

V1-A adds **feature-based personalization** without introducing separate model parameters for each user.

There is still only **one shared global model**.

Personalization comes from adding user-history summary features into the context vector.

So different users get different scores because their history features are different.

## 9. What feature-based personalization means

This is the key idea:

- same model for everyone
- different input features for different users
- therefore, different recommended actions

For example, the model can use features such as:

- this user usually accepts `R` actions
- this user often ignores `O` actions at night
- this user often accepts schedule-related actions in office scenarios

So even with one shared model, behavior can become personalized.

## 10. Recommended user-history feature groups

### A. Global user preference
These summarize the user's overall behavior:

- overall accept rate
- overall deny rate
- recent-N accept rate
- recent consecutive ignore count
- recent consecutive deny count

### B. Action-type preference
Because the main space is R/O, these are very useful:

- accept rate for `R`
- accept rate for `O`
- deny rate for `R`
- deny rate for `O`

### C. Same-scenario preference
These are often the strongest personalization signals:

- accept rate in current scenario
- deny rate in current scenario
- recent feedback score in current scenario
- count of prior exposures in current scenario
- recently accepted action type in current scenario

## 11. V1-A training

Training stays simple:

- one global LinUCB model
- same scenario-based action masking as V0
- base context features + user-history features
- online updates from real user rewards

This is the safest first personalization step.

---

# V1-B: Global LinUCB + Per-User Residual

## 12. Goal

V1-B adds **explicit per-user personalization**.

The idea is:

- keep a shared global model for population-level behavior
- add a small user-specific correction on top

Plain-text formula:

`final_score(user, x, a) = global_score(x, a) + user_residual_score(user, x, a)`

Another plain-text version:

`score(user, x, a) = x · theta_global[a] + x · delta_user[a]`

where:

- `theta_global[a]` is the shared parameter for action `a`
- `delta_user[a]` is the user-specific correction for action `a`

## 13. Why this is better than training one separate model per user

A full separate model per user usually has poor sample efficiency.

Most users will have:

- limited exposure to many scenarios
- limited exposure to many actions
- sparse reward signals

So the better approach is:

- let the global model learn the common patterns
- let the user residual learn only how this user differs from the average

## 14. How to train global + residual

### Global component
Train the global model on data from all users.

This learns the common mapping from context to reward.

### User residual component
For each user:

- initialize residual at zero
- update it only from that user’s own interactions
- regularize it strongly toward zero
- allow it to matter more only after enough user data exists

A good practical rule:

- little user data -> mostly rely on global model
- moderate user data -> mix global and residual
- enough user data -> allow stronger residual influence

## 15. Recommended residual granularity

Do not start with the most detailed version.

Preferred order:

1. user + scenario residual
2. user + action-type residual
3. user + action residual only if data is sufficient

A full user-by-167-action residual table may be too sparse too early.

---

# 16. Suggested rollout path

## Phase 1
Deploy **V0**:

- scenario masking
- strong default prior
- small exploration
- log everything carefully

## Phase 2
Train **V1-A**:

- global LinUCB
- add user-history features
- still one shared model

## Phase 3
Upgrade to **V1-B**:

- keep global model
- add per-user residual
- enable residual more strongly as user data grows

This staged path is safer and easier than jumping directly to fully personalized per-user models.

---

# 17. Final recommendation

A good practical roadmap is:

## V0
**Masked recommendation with strong default prior and small exploration**

- use scenario-based valid-action masking
- make default action win most of the time
- still explore other valid actions occasionally
- synthetic training data should be default-heavy, but not 100% default

## V1-A
**Global LinUCB + user-history features**

- one shared model
- personalization through behavior-summary features

## V1-B
**Global LinUCB + per-user residual**

- shared global backbone
- user-specific correction
- use regularization and data-threshold gating

This path gives:

- safe cold start
- learnable online feedback
- gradual transition to real personalization

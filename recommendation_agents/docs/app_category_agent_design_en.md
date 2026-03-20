# App Category Agent Design Note

## 1. Goal

This agent is for **ranking app categories**, not for recommending a single R/O action.

Given the current state/context, the agent should:
- score 10 app categories
- rank them from most to least relevant
- show the top 6 categories to the user
- learn from exposure order and click feedback

The item placed earlier in the list should be interpreted as more strongly recommended.

## 2. Why this should be a separate agent

This should be a separate agent from the existing R/O bandit, even if both agents share the same state/context features.

Reason:
- the R/O agent recommends **one action**
- the app agent recommends a **ranked list**
- the feedback is different: here we care about **display order, clicked item, click position, and no-click cases**
- the optimization target is different: this is closer to **ranking / slate recommendation** than single-arm recommendation

So the two agents can share:
- context/state features
- user-history features
- logging pipeline
- training infrastructure

But they should have separate:
- action spaces
- rewards
- serving logic
- evaluation metrics

## 3. Recommended roadmap

### V0-App
Use a **contextual scorer + top-k ranking + small exploration**.

This is the recommended first version because it is practical, easy to implement, and good for demo / beta testing.

### V1-App
Add **feature-based personalization** using user history.

### V2-App
If needed later, consider more advanced ranking bandits such as:
- cascading bandit
- position-based bandit
- slate bandit

But these are not recommended for the first version.

## 4. Why not start with cascading bandit

Cascading bandit is related to this problem because users often browse lists from top to bottom and may stop after clicking something.

However, it is not the best first choice because it introduces extra assumptions and complexity:
- assumes users examine items in order
- treats later items as possibly unobserved
- requires more careful exposure and feedback interpretation
- makes implementation and debugging harder

For the first version, it is better to build a simpler and more controllable ranking system first.

## 5. Recommended V0 model

### Core idea
Maintain one scorer for each app category.

For each request:
1. build the context vector
2. score all 10 app categories
3. sort them by score
4. show the top 6
5. log impressions, positions, and click feedback

### Recommended algorithm
Use **scenario-aware top-k LinUCB ranking**.

That means:
- shared state/context features as input
- one LinUCB-style scorer per app category
- compute a score for each category
- rank by score
- display top 6

This is the most natural choice if the team is already using LinUCB for the R/O agent.

## 6. Default app prior

You mentioned that for each scenario you already selected one intuitively best app category.

This is very useful and should be used as a **default prior**.

For cold start, the score can be:

`final_score(category) = model_score(category) + default_bonus(if this category is the scenario default)`

This gives two benefits:
- the list starts from a reasonable scenario-level prior
- other categories can still be explored and eventually outrank the default if data supports it

This is similar to the V0 design for the R/O agent.

## 7. Exploration strategy

The app list should not be fully random, but it should not be fully greedy either.

Recommended approach:
- keep the top 1-2 positions relatively stable
- allow some exploration in positions 3-6

Possible implementations:

### Option A: UCB-based ranking
Use the UCB score for each category and rank by that score.

### Option B: score + controlled shuffle
- keep the highest-scoring 1-2 categories fixed
- sample the remaining displayed categories from the next-best candidates

This is often easier to control in demo settings.

## 8. Feedback and reward design

Each interaction should log:
- shown top-6 categories
- shown order
- clicked category, if any
- clicked position, if any
- no-click case

### Recommended first reward interpretation
If the user clicks category at position `j`:
- clicked category: reward = 1
- items before it: reward = 0 or small negative
- items after it: treat as unobserved for now, not strong negatives

If the user clicks nothing:
- treat the displayed items as 0 for now
- avoid strong negative reward at the beginning

Reason:
- no-click does not always mean bad ranking
- later items may not have been seen
- strong negatives too early can distort learning

## 9. Logging requirements

For every serving event, log:
- user_id
- timestamp
- scenario_id
- context feature vector
- scores for all 10 app categories if feasible
- displayed top-6 categories
- displayed positions
- clicked category
- clicked position
- no-click flag
- model version
- exploration metadata / propensity if feasible

This will make later iteration much easier.

## 10. Feature design

The app agent can reuse the same state/context backbone as the R/O agent.

Recommended feature groups:

### A. Current state/context
- scenario
- time/location/motion/phone/light/sound/day-type
- current state code
- previous state code
- duration features
- battery / charging / mobility / calendar context

### B. User history for apps
Recommended examples:
- overall app-click rate
- recent click rate
- recent no-click count
- click rate in current scenario
- last clicked app category in current scenario
- click counts by app category in current scenario
- click counts by app category overall

This gives feature-based personalization without needing a separate per-user model at the beginning.

## 11. Personalization roadmap

### V1-App: feature-based personalization
Keep one global model, but add user-history features into the context.

This allows the same model to rank differently for different users.

### V2-App: per-user residual personalization
If enough data is collected later, add a user-specific residual on top of the global scorer.

Conceptually:

`score(user, context, category) = global_score(context, category) + user_residual(user, context, category)`

Recommended order of complexity:
1. global model only
2. global model + user history features
3. global model + user residual

Do not jump directly to fully separate per-user models.

## 12. Evaluation

Recommended first metrics:
- click-through rate on displayed top-6
- top-1 click rate
- mean clicked position
- fraction of no-click impressions
- scenario-level click rate
- lift over default-only ranking baseline

If needed later, add ranking metrics such as MRR or NDCG.

## 13. Practical implementation recommendation

For a first working system:
- build a separate App Category Agent
- reuse the same state/context pipeline
- use one scorer per app category
- rank all 10 categories and show top 6
- add a scenario-default app prior
- apply small exploration
- log full impression and click data
- start with a global model
- add user-history features later for personalization

## 14. Final recommendation

The best practical path is:

### V0-App
**Scenario-aware top-k LinUCB ranking + default app prior + small exploration**

### V1-App
**Global ranking model + user-history features**

### V2-App
**Global model + per-user residual**, only after enough real data is available

This gives a stable path from cold start to personalized app ranking without introducing too much complexity too early.

# Next-App Prediction: Paper Deep-Dive

Consolidated notes from a full literature pass covering the 9 papers shortlisted in `next_app_recommendation_report.docx`. For each paper: formulation, architecture, features/modalities, dataset, headline metrics, and stated limitations. Section at the end: cross-paper view of datasets, metrics, and approach families.

---

## 0. Canonical datasets used across the literature

| Dataset | Users | Apps | Records | Span | Location data |
|---|---|---|---|---|---|
| **Tsinghua App Usage** (Shanghai base-station DPI, Apr 2016) | 1,000 | 2,000 | 4,171,950 | 7 days (Apr 19–25) | Base-station ID + POI |
| **LSApp** (Aliannejadi et al.) | 292 | 87 | 3,658,590 (599,635 reduced) | ~15 days (2018) | None |
| **China Mobile — Shanghai / Nanchang** | operator-scale | — | tens of millions | ~1 week | Cell tower |
| **Shanghai Telecom App-usage trace** (Appformer) | ~10k | — | ~8.3M | 1 week, Apr 20–26 2016 | Base station + 17-d POI |
| **Operator dataset** used by ATPP | 443 | — | 2,104,432 | 21 days | Cell ID |
| **Various commute-focused logs** (AppUsageOTM) | unclear | — | — | — | Location trajectory |

Most of the newer papers (MISApp, TGT, MAPLE, Chen-Scientific Reports, Dynamic Hypergraphs) converge on **Tsinghua App Usage** as the primary benchmark, typically paired with LSApp for location-free evaluation.

---

## 1. MISApp — Multi-Hop Intent-Aware Session Graph Learning (2026)

- **Citation.** Li, Qu et al., arXiv:2603.21653, March 2026. *Profile-free* next-app prediction.
- **Problem.** Given a session S = [a_1..a_T], time context, base-station context → rank next app. Session cutoff at 300 s idle. T fixed at 8.
- **Architecture (key insight: multi-hop session graph):**
  - Build **three graphs** per session: 1-hop (consecutive transitions), 2-hop (compositions), 3-hop (longer co-usage).
  - Each graph → **LightGCN propagation** (no transform weights), layer-average → query-attention pool using the last app as query → a per-hop session vector g^(γ).
  - **Hop-level softmax attention** fuses g^(1..3) into a single graph embedding g.
  - **Temporal stream**: 24-bucket hour lookup. **Spatial stream**: base stations clustered by POI cosine similarity, embedded.
  - **Cross-Modal Gated Fusion (CMGF)**: multi-head sigmoid gates apply cross-attention between graph / time / location. Output h.
  - **Transformer encoder** over h; MLP intent vector s from last K=3 apps; **Transformer decoder** cross-attends to h with s as query; decoder output u → softmax logits.
  - Loss: cross-entropy.
- **Features.** App ID embeddings, hour, POI-clustered base station. **No user embedding** (this is the profile-free part).
- **Results (Tsinghua / LSApp):** ACC@1 **0.5424 / 0.8394**, ACC@5 **0.8277 / 0.9668**, MRR@5 **0.6438 / 0.8790**. Cold-start Tsinghua ACC@1 **0.5463** (vs FEDformer 0.5296).
- **Ablation take-away.** Removing the intent-decoder is the most damaging (-8.7 pp ACC@1); removing multi-hop graph, time, or spatial each cost ~2-2.3 pp.
- **Limitations.** Only two datasets; fixed window 8; no on-device latency discussion.
- **URL.** https://arxiv.org/abs/2603.21653

## 2. TGT / Atten-Transformer — Temporal Gating Transformer (2025)

- **Citation.** Li, Qu, Wang. arXiv:2502.16957 (v1 Feb 2025 "Atten-Transformer"; v2 Sep 2025 retitled "TGT").
- **Problem.** Session of recent app events + hour-of-day → predict next app.
- **Architecture.** Transformer over app events, **temporal gating module** between layers: hour τ is Fourier-encoded to P_h, then `g_t = σ(W_h P_h)` acts as an elementwise gate on the pooled session representation. This is *orthogonal* to self-attention — not a positional encoding, a feature-level scale. Also carries a user-profile embedding fused pre-gate.
- **Features.** App ID, per-app duration, hour (Fourier), user-profile ID. No explicit location.
- **Results.** Tsinghua HR@1 **0.527**, HR@5 **0.813**, MRR@5 **0.636**. LSApp HR@1 **0.811**, HR@5 **0.965**, MRR@5 **0.878**. Gains over MAPLE are modest (~0.7–0.9 pp HR@1) but cold-start delta is larger.
- **Ablation highlight.** Removing temporal gating loses **−18.6 % HR@1** — temporal gating is the dominant contribution. Fourier > MLP > RBF for time encoding.
- **Limitations.** Lacks location/POI, fixed user vocabulary — can't generalize to unseen users without re-training the profile embedding.
- **URL.** https://arxiv.org/abs/2502.16957

## 3. Chen et al. 2025 — GGNN + Self-Attention (Scientific Reports)

- **Citation.** Chen, Yu, Zhao et al. *Sci. Rep.* 15: 2025, DOI 10.1038/s41598-025-05260-1 (also PMC12216173).
- **Problem.** Session-level next-app ranking using Tsinghua data; session split at idle.
- **Architecture.** Two branches, late fusion:
  - **Short-term:** per-session directed graph (node = app, weighted transition edges A_in / A_out) → **GGNN** (Gated Graph Neural Net, k=3 iterations) produces node states; soft-attention pool (last app as query) → session vector `v_s`.
  - **Long-term:** all recent app events → multi-head self-attention (2 stacked blocks, 8 heads) → user preference vector `v_l`.
  - **Fusion.** Compared addition, concat, and **soft-attention gate** — soft-attention wins. Output head: `ŷᵢ = softmax(v · e_i)`.
  - Auxiliary 100 m×100 m grid location embedding, hour and weekday features.
- **Features.** App ID, hour/weekday, base-station 100 m grid, user ID.
- **Results (Tsinghua).** HR@5 **0.8277**, HR@10 **0.8778**, HR@20 **0.8641**; MRR@5 **0.4298**, NDCG@5 **0.5206**. Headline improvement ~+11.9 % HR@1 over GCSAN, statistically significant (p<0.05, paired-t).
- **Limitations.** Single week, single dataset (Tsinghua), no user demographics, cold-start not evaluated.
- **URL.** https://www.nature.com/articles/s41598-025-05260-1

## 4. Appformer (2024) — Progressive multimodal fusion

- **Citation.** Sun et al., arXiv:2407.19414; DOI 10.1016/j.eswa.2024.125903 (Expert Systems with Applications).
- **Problem.** Given m=4 recent (app, time, POI, user) events, predict next app.
- **Architecture.** Two stages:
  - **Progressive multimodal fusion** (Transformer-decoder-style block): app+time first, then + user, then + POI. Each step uses masked self-attention + cross-attention to already-fused tensor. This avoids forcing all modalities into one concat at input and lets the model prioritize app/time as the backbone.
  - **Feature-extraction encoder-decoder Transformer**: encoder sees the fully fused tensor; decoder sees only app+time tensor (keeps decoding channel narrow). Final head → softmax over app vocab.
  - K-Modes clustering on POI vectors (17-d) outperformed K-Means / K-Means++ / K-Medoids.
- **Features.** App ID embedding, time-of-day embedding, 17-d POI (K-Modes→bucket), user ID embedding.
- **Results.** Shanghai Telecom: HR@1 **31.92 %** (partition 1), HR@1 **42.68 %** (partition 2). Outperforms DUGN, AppUsage2Vec, DeepApp by 4–8 pp HR@1.
- **Limitations.** Limited personalization (small user vocab); Transformer cost non-trivial for on-device; no cold-start treatment.
- **URL.** https://arxiv.org/abs/2407.19414

## 5. MAPLE (2024) — LLM embeddings + installed-app cold-start

- **Citation.** Khaokaew, Xue, Salim. *Proc. ACM IMWUT* 8(1) Art. 10, Mar 2024. DOI 10.1145/3643514. arXiv:2309.12451.
- **Problem.** Next-app prediction reformulated as **conditional text generation**: context templated into natural-language sentences; a seq2seq LLM emits a ranked list of app names.
- **Architecture.** Two-stage LLM pipeline (T5-large backbone):
  - **Stage 1 — ATP (App Type Prediction):** prompt includes recent apps, times, POIs; LM emits the next *category* token(s).
  - **Stage 2 — Next-app decoding:** prompt now includes Stage-1 output; LM emits the next *app* token(s).
  - **Cold-start injection:** for new users, find nearest existing user by Jaccard similarity of installed-app sets and use their history as prompt seed.
  - Training: standard cross-entropy over seq2seq; Adam, batch 8, LR 1e-5, 30 epochs.
- **Features.** App name strings (tokenized by LM), POI names, hour, installed-app list, user installed-app set.
- **Results (Tsinghua / LSApp):**
  - Regular: ACC@1 **0.5191 / 0.4920**, ACC@5 **0.8064 / 0.8128**.
  - Cold-start: MAPLE still wins by ~0.02–0.06 ACC@1 over LLM baselines.
- **Limitations.** LLM inference cost; no explicit session-graph modeling; POI/time handled only via templated strings.
- **URL.** https://dl.acm.org/doi/10.1145/3643514

## 3. Chen et al. (Scientific Reports, 2025) — (already covered above as paper 3)

## 6. Context-Aware Dynamic Hypergraphs (IEEE TMC 2025)

- **Citation.** Huang, Li, Wang et al. *IEEE Trans. Mobile Comput.* 24(1), 2025. DOI 10.1109/TMC.2024.3374551.
- **Problem.** Treat apps within a session as a **hyperedge** (higher-order relation over arbitrary subsets), not pairwise edges. Sequence of hyperedges evolves over time → dynamic hypergraph.
- **Architecture.**
  - **Intra-session hypergraph convolution**: HGCN aggregates apps within a session hyperedge → session intent vector.
  - **Inter-session Transformer**: stacks session intents, self-attention across recent sessions for evolution.
  - **Context injection**: hour/weekday/POI embeddings concatenated to app embeddings before HGCN.
  - MLP head over fused (latest intent) + (last-app embedding) → softmax.
- **Features.** App, time, location/POI, session boundary, user ID.
- **Datasets.** China Mobile **Shanghai** and **Nanchang** operator traces (tens of thousands of users; several million records).
- **Results.** Relative improvements **>30 %** on Shanghai and **~20 %** on Nanchang over AppUsage2Vec / DeepApp / DUGN / CAP baselines (absolute HR@1 numbers behind IEEE paywall, not independently verified).
- **Limitations.** Hypergraph memory grows with session count; operator-specific datasets not public; privacy-sensitive.
- **URL.** https://ieeexplore.ieee.org/document/10844996

## 6. Context-Aware Dynamic User Graph Network — DUGN (IEEE TMC 2023)

- **Citation.** Ouyang, Guo, Wang, Liang, Li. *IEEE Trans. Mobile Comput.* 22(8) 2023, DOI 10.1109/TMC.2022.3161114. (Often cited as the **first "dynamic app-usage graph"** baseline.)
- **Problem.** Given recent sessions, recommend next app.
- **Architecture.** Per-session app co-usage graph → node-level GAT → graph-level attention → session vector → RNN over session vectors captures dynamic preference → softmax next-app head.
- **Features.** App IDs, transition / co-usage edges, session times.
- **Results.** Reported SOTA on Tsinghua at time of publication; exact HR@k numbers behind IEEE paywall. Cited as a consistent baseline by MISApp, Appformer, Dynamic Hypergraph.
- **Limitations.** No location/POI fusion in the original model; pairwise edges cannot capture higher-order co-usage (this is the gap MISApp / Dynamic Hypergraph address).
- **URL.** https://ieeexplore.ieee.org/document/9739939

## 7. ATPP (2021 / 2023) — Marked Temporal Point Process

- **Citation.** Yang, Wang, Zhao, Zhang, Zou, Du. DCOSS 2021 Best Paper Runner-up; extended *ACM TOSN* 19(3) 2023 (DOI 10.1145/3582555).
- **Problem.** Jointly predict *which* app and *when* it will be opened. Mark = app, time = seconds-since-previous-event.
- **Architecture.**
  - **App-embedding** via skip-gram over co-usage sequences.
  - **Attention RNN history encoder** produces hidden h_t after every event.
  - **Feature fusion**: concat app-emb, location emb, hour-of-day to h_t.
  - **MTPP head**: conditional intensity `λ*(t | H)` as a neural function; separate **mark distribution** head (softmax over apps).
  - Loss = log-likelihood of (t_i, m_i) = time NLL + mark CE.
  - Inference: argmax mark; expected time via closed-form integral of intensity.
- **Features.** App ID, inter-event time, cell-location.
- **Results (operator data, 443 users, 21 days).** ACC@1 up to **81.5 %**; time MAE **~1.45 min**. Applied downstream to *prefetching*, authors claim **78.1 %** app-launch latency reduction.
- **Limitations.** Small user base; no semantics; MTPP head costs stability on cold users.
- **URL.** https://dl.acm.org/doi/10.1145/3582555

## 7. AppUsageOTM (Kang et al., 2022) — Commute-aware

- **Citation.** Kang, Rahaman, Ren, Sanderson, Sadri. *Pervasive & Mobile Computing* 87, 2022, DOI 10.1016/j.pmcj.2022.101704.
- **Problem.** Same-day *commute context* next-app prediction: capture that users switch app intent during different stages of their commute.
- **Architecture.** Two-stage:
  1. **Commute-stage classifier** — HMM/RF over sensor + location windows that labels each event with a commute-stage (leaving home, in transit, arriving work, etc.).
  2. **Next-app ranker conditioned on stage** — GRU/Transformer over recent app events, concatenating stage one-hot and contextual features (time, location). Softmax head over candidate apps.
- **Features.** App IDs, commute-stage labels, GPS-derived trajectory features (speed, acceleration), time-of-day, day-of-week.
- **Results.** HR@1 improvement of ~**5-10 %** over context-less baselines on commute periods; smaller gains outside commute windows.
- **Limitations.** Requires GPS/sensor access; only meaningful when a commute exists (otherwise upstream classifier degrades); small datasets.
- **URL.** https://doi.org/10.1016/j.pmcj.2022.101704

---

## Cross-paper view

### Approach families (rolling up)

| Family | Papers | Key primitive | Handles higher-order? | Handles time? | Cold-start? |
|---|---|---|---|---|---|
| Session graph / hypergraph | MISApp, Dynamic Hypergraph, DUGN, Chen 2025 | HGCN / GGNN / GAT on per-session graph | ✅ (multi-hop / hyperedge) | Partial (stream over sessions) | MISApp only |
| Temporal transformer | TGT / Atten-Transformer | Fourier time + gating | ❌ (sequence only) | ✅ (core contribution) | ❌ |
| Multimodal fusion | Appformer, AppUsageOTM | Cross-attention / K-Modes POI / commute stage | ❌ | Partial | ❌ |
| Semantic LLM | MAPLE | Seq2seq LLM + installed-app similarity | Implicit via prompt | Partial (hour in template) | ✅ (Jaccard bootstrap) |
| Temporal point process | ATPP | Conditional intensity + mark | ❌ | ✅ (joint time + mark) | ❌ |

### Headline numbers on Tsinghua App Usage (standard split)

| Model | ACC@1 / HR@1 | ACC@5 / HR@5 | MRR@5 | Uses location |
|---|---|---|---|---|
| MFU / MRU (baselines) | ~0.33–0.42 | ~0.68–0.74 | ~0.48 | No |
| AppUsage2Vec | 0.466 | 0.779 | 0.602 | No |
| NeuSA / NeuSA+ | 0.464 | 0.729 | 0.580 | No |
| MAPLE | **0.519** | 0.806 | 0.633 | Partial |
| TGT | **0.527** | 0.813 | 0.636 | No |
| Chen 2025 GGNN+SA | 0.3–0.5 reported on per-event split | 0.665 | 0.430 | ✅ 100 m grid |
| MISApp | **0.542** | **0.828** | **0.640** | ✅ POI-clustered |

(Chen 2025 uses a per-event evaluation split rather than the session-oriented split used by MISApp/TGT/MAPLE, so the numbers are *not* directly comparable.)

### Cross-cutting patterns

1. **Sessionization is the foundation** — everything above splits on idle gap (~5 min). The session is the unit of "intent."
2. **Higher-order co-usage beats pairwise** — MISApp's multi-hop graph, Dynamic Hypergraph's hyperedges, and Chen 2025's k-hop GGNN all outperform pairwise transition models on the same data.
3. **Explicit temporal conditioning matters** — TGT shows a 18.58 % HR@1 collapse when you remove the temporal gate. Hour-of-day and weekday embeddings are the dominant time features.
4. **Location is worth including when you have it** — MISApp, Appformer, Dynamic Hypergraph, AppUsageOTM all extract gains from POI / cell tower. But raw cells are noisy — clustering by POI content (MISApp, Appformer's K-Modes) cleans it up.
5. **Cold-start is still not well-solved** — only MAPLE (installed-app Jaccard) and MISApp (profile-free architecture) make cold-start a first-class design constraint.
6. **Dataset monoculture** — Tsinghua and LSApp dominate. Headline numbers from operator datasets (Dynamic Hypergraph, AppUsageOTM) are less directly comparable.

### What a "modern" next-app predictor would look like, synthesizing the above

1. **Ingest**: stream of (user, app, timestamp, coarse-location) events.
2. **Sessionize**: 5-min idle cutoff → bounded session of 8–16 events.
3. **Build a multi-hop session graph** (1-hop, 2-hop, 3-hop) with query-attention pooling (MISApp) or a session hyperedge (Dynamic Hypergraph).
4. **Fuse time**: Fourier-encoded hour-of-day, weekday, elapsed since last app; gated into session rep (TGT).
5. **Fuse location**: POI-clustered cell embedding (MISApp / Appformer K-Modes).
6. **Cold-start backup**: installed-app Jaccard lookup to seed embedding for new users (MAPLE).
7. **Optional time prediction**: MTPP intensity head if prefetching is a goal (ATPP).
8. **Optional long-term channel**: long-range self-attention over user history, fused by gate into the session representation (Chen 2025).

This is the "hybrid stack" the report recommends, now with concrete primitives from each family.

### Open questions for our task

- **Do we have user IDs?** Determines whether we can use any profile channel at all; if this is a session-only problem we should follow MISApp's profile-free design.
- **Do we have location / POI?** If not, rules out a whole modality; if yes, cell-tower clustering is almost always beneficial.
- **Do we care about "when" or only "what"?** If yes (preload, notifications), ATPP-style MTPP becomes necessary.
- **What is the evaluation protocol?** Per-event vs per-session splits differ by up to ~20 pp — any benchmarking we do needs to fix this up front.
- **Are we constrained to on-device inference?** LLM-based MAPLE is probably too heavy; Transformer session models (TGT) are the lightest high-performers.

Open these before any modeling choice — each one would substantially narrow the design space.

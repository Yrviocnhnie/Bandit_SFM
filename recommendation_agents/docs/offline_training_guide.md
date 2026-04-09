# Offline Training Guide

这份文档说明如何从 raw JSONL 数据出发，跑出当前这版最佳 artifact：

- [v6_2k_neural_linear_3p3a_hardneg_other_stratified_split](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split)

这里的模型是：

- `NeuralEncoder LinUCB (UCB random init)`
- CLI 里的 `--model-type neural-linear`

默认假设：

- 当前工作目录是仓库根目录 `Bandit_SFM/`
- 也就是你运行命令时所在目录包含：
  - `data/`
  - `recommendation_agents/`

## 1. 最终产物

训练完成后，会得到一个目录，例如：

- `artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split`

里面主要包括：

- `train.raw.jsonl`
- `test.raw.jsonl`
- `ro_train_samples_expanded.jsonl`
- `app_train_samples_expanded.jsonl`
- `ro_model/`
- `app_model/`
- `eval_both_top3.json`
- `train_both_summary.json`
- `v6_training_config.json`

## 2. 需要什么数据

### 2.1 Raw input JSONL

当前这版最佳模型使用的 raw 数据是：

- `data/bandit_v0_65_scenaroios_unique_support_to_2k_samples_office_working_alias.jsonl`

每一行是一个 raw context sample，包含至少这些字段：

- `scenario_id`
- `episode_id`
- `features`

这份数据里虽然也有：

- `gt_ro`
- `gt_app`

但在 `run-v6-plan-b` 这条 workflow 里，**这些 ground-truth 不会直接拿来训练**。

训练时真正使用的是：

- `scenario_id`
- `features`
- `recommendation_agents/docs/scenario_recommendation_actions_v6.md` 里对每个 scenario 定义的：
  - `most_relevant`
  - `plausible`
  - `irrelevant`

也就是说，训练阶段会根据 `scenario_id`，把每个 raw context 展开成全 action space 的训练样本。

### 2.2 Relevance catalog

需要这份 scenario-to-action 标注表：

- [scenario_recommendation_actions_v6.md](scenario_recommendation_actions_v6.md)

训练时就是根据这份 markdown，把每个 raw context 扩展成：

- R/O: `46` 个 action 样本
- App: `10` 个 category 样本

### 2.3 Global action catalog

还需要 action catalog markdown：

- [scenario_recommendation_actions_v5.md](scenario_recommendation_actions_v5.md)

这份文件用于构建：

- `ro_metadata.json`
- `app_metadata.json`

## 3. 这版数据为什么用 alias 文件

原始 65-scenario 数据里，有一个 scenario 名字是：

- `OFFICE_AFTERNOON`

但当前 relevance catalog 里对应的是：

- `OFFICE_WORKING`

为了和现有 `v6` 标注表对齐，这次训练用的是一个 alias 版输入文件，把：

- `OFFICE_AFTERNOON -> OFFICE_WORKING`

如果你手里只有原始文件：

- `data/bandit_v0_65_scenaroios_unique_support_to_2k_samples.jsonl`

可以先运行下面这个脚本生成 alias 文件：

```bash
python - <<'PY'
import json
from pathlib import Path

src = Path('data/bandit_v0_65_scenaroios_unique_support_to_2k_samples.jsonl')
dst = Path('data/bandit_v0_65_scenaroios_unique_support_to_2k_samples_office_working_alias.jsonl')

with src.open() as fin, dst.open('w') as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('scenario_id') == 'OFFICE_AFTERNOON':
            row['scenario_id'] = 'OFFICE_WORKING'
            if row.get('scenario_name') == 'Office afternoon':
                row['scenario_name'] = 'Office working'
        fout.write(json.dumps(row, sort_keys=True))
        fout.write('\\n')

print(dst)
PY
```

## 4. 训练入口

当前推荐使用 CLI：

- `python -m recommendation_agents.cli run-v6-plan-b`

对应实现入口在：

- [cli.py](../recommendation_agents/cli.py)
- [workflows.py](../recommendation_agents/workflows.py)

`run-v6-plan-b` 会自动做这些事：

1. 读 raw input JSONL
2. 按 `scenario_id` 做 `80/20 scenario-stratified split`
3. 写出：
   - `train.raw.jsonl`
   - `test.raw.jsonl`
4. 构建：
   - `ro_metadata.json`
   - `app_metadata.json`
5. 根据 `scenario_recommendation_actions_v6.md` 扩展训练样本：
   - `ro_train_samples_expanded.jsonl`
   - `app_train_samples_expanded.jsonl`
6. 训练：
   - `ro_model/`
   - `app_model/`
7. 在 test split 上评估并输出：
   - `eval_both_top3.json`

## 5. 训练配置

当前最佳模型使用的是：

- model type: `neural-linear`
- split: `Plan B`
- train/test split: `80/20 scenario-stratified`
- reward config:
  - `most_relevant = 1.0`
  - `plausible = 0.1`
  - `irrelevant = -0.1`
  - `other = 0.0`

这个配置在产物目录里也会保存到：

- [v6_training_config.json](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/v6_training_config.json)

`other = 0.0` 是通过：

- `--other-zero-mode exclude-all-labeled`

实现的，含义是：

- 已经被标成 `most_relevant / plausible / irrelevant` 的 action 保持原 reward
- 其余未标注 action 自动补成 `0 reward`

所以对每个 raw context：

- R/O 最终总共会展开成 `46` 条训练样本
- App 最终总共会展开成 `10` 条训练样本

## 6. 一条命令跑出这版最佳模型

如果你想从 raw 数据直接重新跑出这版 artifact，用这条命令：

```bash
conda run --no-capture-output -n sfm python -m recommendation_agents.cli run-v6-plan-b \
  --input data/bandit_v0_65_scenaroios_unique_support_to_2k_samples_office_working_alias.jsonl \
  --catalog-markdown recommendation_agents/docs/scenario_recommendation_actions_v5.md \
  --output-dir recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split \
  --relevance-markdown recommendation_agents/docs/scenario_recommendation_actions_v6.md \
  --test-ratio 0.2 \
  --most-relevant-reward 1.0 \
  --plausible-reward 0.1 \
  --irrelevant-reward -0.1 \
  --most-relevant-repeat 1 \
  --plausible-repeat 1 \
  --irrelevant-repeat 1 \
  --other-zero-mode exclude-all-labeled \
  --alpha 0.0 \
  --default-bonus 0.0 \
  --epochs 1 \
  --top-k 3 \
  --progress-every 10000 \
  --device cpu \
  --model-type neural-linear \
  --no-track-train-hit-rate
```

如果你希望用 GPU，可以把：

- `--device cpu`

改成：

- `--device cuda`

## 7. 训练完成后如何确认结果

重点看这几个文件：

- [train_both_summary.json](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/train_both_summary.json)
- [eval_both_top3.json](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/eval_both_top3.json)
- [ro_model/manifest.json](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/ro_model/manifest.json)
- [app_model/manifest.json](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/app_model/manifest.json)

这版最佳模型对应的离线评估结果是：

- `R/O`: `2.90 / 2.95 / 0.01`
- `App`: `2.92 / 2.96 / 0.01`

## 8. 输出目录里几个关键文件是什么意思

### `train.raw.jsonl`

原始训练 split，每行一个 raw context。

### `test.raw.jsonl`

原始测试 split，每行一个 raw context。

### `ro_train_samples_expanded.jsonl`

给 R/O 模型训练用的展开样本。

含义是：

- 每个 raw training context 会复制成 `46` 条
- 每条对应一个具体 R/O action
- 并附上该 action 的 reward

所以这个文件会很大，这是正常的。

### `app_train_samples_expanded.jsonl`

给 App 模型训练用的展开样本。

- 每个 raw training context 会复制成 `10` 条
- 每条对应一个具体 app category

### `ro_model/` 和 `app_model/`

训练好的模型 artifact。

对 `neural-linear` 来说：

- encoder 是 `306 -> 128 -> 64`
- 再加上每个 action 一个 disjoint LinUCB head

## 9. 相关文档

- 结果汇总：
  - [report_2026_04_01.md](report_2026_04_01.md)
- 训练数据的 scenario-to-action 标注：
  - [scenario_recommendation_actions_v6.md](scenario_recommendation_actions_v6.md)
- action catalog：
  - [scenario_recommendation_actions_v5.md](scenario_recommendation_actions_v5.md)

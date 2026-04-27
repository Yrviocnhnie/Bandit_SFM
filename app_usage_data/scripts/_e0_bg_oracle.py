"""E0 sanity: compute BG-state oracle bounds on test split."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from lib.v3.bg_state import reconstruct_bg

vocab = json.load(open(ROOT / "artifacts/vocab.json"))
V = len(vocab)
test = pd.read_parquet(ROOT / "artifacts/splits/test.parquet")
snaps = reconstruct_bg(test, vocab)
n = len(snaps)

# Hit@1 oracle: pick the most recently used app in BG
correct = 0
for s in snaps:
    rec = np._unused = None
import numpy as np
correct = 0
empty_bg = 0
for s in snaps:
    rec = np.where(s.bg_mask, s.bg_recency_sec, np.inf)
    if np.isfinite(rec).any():
        a = int(rec.argmin())
        if a == s.target_app:
            correct += 1
    else:
        empty_bg += 1

print(f"snapshots:       {n}")
print(f"is_reuse rate:   {sum(s.is_reuse for s in snaps)/n:.3f}  (target was already in BG)")
print(f"BG-MRU Hit@1:    {correct/n:.3f}   <-- predict 'most recently used in BG'")
print(f"empty BG events: {empty_bg}/{n}")
print(f"  current v3 R6 Hit@1 = 0.599 (multi-headed model)")
print()

# EH@5 oracle: top-5 by recency from BG
target_t = test[test['is_target_event'].astype(bool)].sort_values('event_ts').reset_index(drop=True)
ts = pd.to_datetime(target_t['event_ts']).to_numpy().astype('datetime64[ns]').astype(np.int64)
labs = target_t['app_label_clean'].astype(str).tolist()
WIN = 900 * 10**9
te_total = te_hit = 0
for i, s in enumerate(snaps):
    end = ts[i] + WIN
    rec_v = np.where(s.bg_mask, s.bg_recency_sec, np.inf)
    order = np.argsort(rec_v)
    top5 = set(int(x) for x in order[:5] if s.bg_mask[x])
    j = i
    while j < n and ts[j] <= end:
        a = vocab.get(test_t_labels(j) if False else target_t['app_label_clean'].iloc[j], vocab.get('<RARE>', 2))
        te_total += 1
        if a in top5:
            te_hit += 1
        j += 1
print(f"BG-top5-by-recency EH@5: {te_hit/max(1,te_total):.3f}  (events {te_total}, hits {te_hit})")
print(f"  current v3 R6 EH@5 = 0.746")

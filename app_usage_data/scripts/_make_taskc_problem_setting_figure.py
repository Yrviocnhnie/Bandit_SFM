"""Render the Task C problem-setting diagram (2 h background / 1 h prediction).

Run from anywhere; output goes to ../figures/taskc_problem_setting.png
(relative to this script).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = str(ROOT / "figures" / "taskc_problem_setting.png")
(ROOT / "figures").mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13.5, 6.6), dpi=140)

# ===== Time coordinates (decimal hours) =====
T_LEFT, T_BG_START, T_ANCHOR, T_END = 11.5, 12.0, 14.0, 15.0  # 11:30 ... 15:00

# ===== Top windows =====
y_win = 1.55
ax.add_patch(Rectangle((T_BG_START, y_win), T_ANCHOR - T_BG_START, 0.40,
                       facecolor="#cfe2f3", edgecolor="#3d85c6", linewidth=1.4))
ax.text((T_BG_START + T_ANCHOR) / 2, y_win + 0.20,
        "BACKGROUND WINDOW (2 h)\nlook back from anchor t — collect B(t)",
        ha="center", va="center", fontsize=10, color="#1a3a55")

ax.add_patch(Rectangle((T_ANCHOR, y_win), T_END - T_ANCHOR, 0.40,
                       facecolor="#ffe599", edgecolor="#bf9000", linewidth=1.4))
ax.text((T_ANCHOR + T_END) / 2, y_win + 0.20,
        "PREDICTION WINDOW (H = 60 min)\npredict y_used per app in B(t)",
        ha="center", va="center", fontsize=10, color="#5a4500")

# ===== Anchor vertical line =====
ax.plot([T_ANCHOR, T_ANCHOR], [0.05, 1.95],
        color="#990000", linewidth=2.0, linestyle="--", alpha=0.85)
ax.text(T_ANCHOR, 2.02, "anchor t = 14:00",
        ha="center", va="bottom", fontsize=11, fontweight="bold", color="#990000")

# ===== Timeline axis =====
y_axis = 1.30
ax.plot([T_LEFT, T_END], [y_axis, y_axis], color="black", linewidth=1.6)
for x, label in [(T_LEFT, "11:30"),
                 (T_BG_START, "12:00\n(t − 2h)"),
                 (T_ANCHOR, "14:00\n(anchor)"),
                 (T_END, "15:00\n(t + 1h)")]:
    ax.plot([x, x], [y_axis - 0.04, y_axis + 0.04], color="black", linewidth=1.4)
    ax.text(x, y_axis - 0.10, label, ha="center", va="top",
            fontsize=9, color="black",
            fontweight=("bold" if "(anchor)" in label else "normal"))

# ===== Per-app rows =====
# (label, FG intervals, return time in [T_ANCHOR, T_END] or None, in B(t)?)
apps = [
    ("App A", [(11.50, 12.25)], 14.583, True,  "14:35"),  # 11:30→12:15 FG, 14:35 FG
    ("App B", [(13.083, 13.50)], None,    False, None),   # 13:05→13:30 FG, no return
    ("App C", [(13.75, 13.917)], 14.333, True,  "14:20"), # 13:45→13:55 FG, 14:20 FG
]

ROW_TOP = 0.92
ROW_H = 0.22
for i, (name, fg_intervals, ret_t, in_bt, ret_label) in enumerate(apps):
    y = ROW_TOP - i * ROW_H

    # row baseline
    ax.plot([T_LEFT, T_END], [y, y], color="#bbbbbb", linewidth=0.6, linestyle=":")
    ax.text(T_LEFT - 0.10, y, name, ha="right", va="center",
            fontsize=11, fontweight="bold")

    # foreground bars (dark blue)
    for (a, b) in fg_intervals:
        ax.add_patch(Rectangle((a, y - 0.045), b - a, 0.09,
                               facecolor="#3d85c6", edgecolor="black", linewidth=0.8))
        ax.text((a + b) / 2, y + 0.075, "FG",
                ha="center", va="bottom", fontsize=8,
                color="#1a3a55", fontweight="bold")

    # background segment from end-of-FG to anchor (light blue stripe)
    last_fg_end = fg_intervals[-1][1]
    ax.add_patch(Rectangle((last_fg_end, y - 0.030), T_ANCHOR - last_fg_end, 0.06,
                           facecolor="#cfe2f3", edgecolor="#3d85c6",
                           linewidth=0.6, alpha=0.85))
    # Label INSIDE the BG stripe (not below it) to avoid collisions with the
    # next row's FG label.
    if T_ANCHOR - last_fg_end > 0.40:    # only label if there's room
        ax.text((last_fg_end + T_ANCHOR) / 2, y, "background",
                ha="center", va="center", fontsize=8,
                color="#1a3a55", style="italic")

    # return event (red triangle pointing up) or "no return" marker
    if ret_t is not None:
        ax.plot(ret_t, y, marker="^", color="#cc0000", markersize=11, zorder=5)
        ax.text(ret_t, y + 0.10, f"FG\n{ret_label}",
                ha="center", va="bottom", fontsize=8.5,
                color="#cc0000", fontweight="bold")
    else:
        # explicit "no FG in window" — gray X near the right end
        ax.text((T_ANCHOR + T_END) / 2, y + 0.06,
                "✗  no FG event in (t, t+1h]",
                ha="center", va="bottom", fontsize=9,
                color="#7f7f7f", style="italic")

    # truth label on the right
    if in_bt:
        truth = "y_used = 1\n(KEEP — user came back)"
        col = "#226633"
    else:
        truth = "y_used = 0\n(SAFE TO KILL)"
        col = "#993333"
    ax.text(T_END + 0.12, y, truth, ha="left", va="center",
            fontsize=9.5, color=col, fontweight="bold")

# ===== B(t) annotation, in its own dedicated row below =====
y_bt = ROW_TOP - 3 * ROW_H + 0.02
ax.text(T_LEFT, y_bt, "B(t) = { A, B, C }",
        ha="left", va="center", fontsize=12, fontweight="bold", color="#1a3a55",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="#fffacd", edgecolor="#bf9000", linewidth=1.2))
ax.text(T_LEFT + 1.2, y_bt,
        "= apps backgrounded inside the 2 h window — model must score each",
        ha="left", va="center", fontsize=9.5, color="#1a3a55", style="italic")

# ===== Title + axis hygiene =====
ax.set_title("Task C — Background-app suspension prediction\n"
             "anchor t · backward window 2 h (build B(t)) · forward window H = 60 min (label y_used)",
             fontsize=12.5, pad=12)
ax.set_xlim(11.0, 16.0)
ax.set_ylim(0.18, 2.18)
ax.set_xticks([])
ax.set_yticks([])
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"wrote {OUT}")

"""Background-app state machine for Task C.

Walks the chronologically-sorted event stream and emits, for each anchor
time `t`, the set `B(t)` of apps currently in background plus per-app
scalars (time in background, time since last foreground, fg_count_today,
last foreground loc_id and daypart bin).

Event semantics (counts from the `final_compact` sheet):
    APP_FOREGROUND / APP_START  -> status = "FG"
    APP_BACKGROUND              -> status = "BG"
    PROCESS_EXIT                -> status = "DEAD"
    PROCESS_START               -> records last_start_ts (no status change)
    everything else             -> ignored

Staleness cutoff: any BG app with `now - last_bg_ts` greater than
`T_STALE_SEC` (default 6 h) is treated as DEAD at snapshot time. This
mirrors real-device eviction and keeps |B(t)| bounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


FG_NAMES = frozenset({"APP_FOREGROUND", "APP_START"})
BG_NAMES = frozenset({"APP_BACKGROUND"})
EXIT_NAMES = frozenset({"PROCESS_EXIT"})
START_NAMES = frozenset({"PROCESS_START"})
SCREEN_ON_NAMES = frozenset({"SCREENON_EVENT"})
SCREEN_OFF_NAMES = frozenset({"SCREENOFF_EVENT"})

T_STALE_SEC_DEFAULT = 6 * 3600
KILL_RECENT_WINDOW_SEC = 10 * 60   # how long we treat a PROCESS_EXIT as "recent"
NS_PER_SEC = 1_000_000_000
NS_PER_DAY = 86_400 * NS_PER_SEC


@dataclass
class AppState:
    status: str = "DEAD"
    last_fg_ts_ns: int = -1
    last_bg_ts_ns: int = -1
    last_start_ts_ns: int = -1
    fg_count_today: int = 0
    last_day_idx: int = -1
    last_fg_loc_id: int = 0
    last_fg_daypart: int = -1


@dataclass
class BGSnapshot:
    anchor_ts_ns: int
    apps: List[str] = field(default_factory=list)
    time_in_bg_sec: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    time_since_fg_sec: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    fg_count_today: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    last_fg_loc_id: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    last_fg_daypart: np.ndarray = field(default_factory=lambda: np.full(0, -1, dtype=np.int32))
    last_fg_app: Optional[str] = None
    # NEW for v2: anchor-level scalars (broadcast to all rows in the anchor)
    time_since_screen_on_sec: float = -1.0
    last_kill_app: Optional[str] = None     # most recent PROCESS_EXIT'd app (within KILL_RECENT_WINDOW_SEC)
    last_kill_age_sec: float = -1.0          # seconds since that kill; -1 if no recent kill
    bg_recency_min_sec: float = -1.0         # min time_in_bg_sec across B(t); -1 if empty


def _ensure(state, app):
    s = state.get(app)
    if s is None:
        s = AppState()
        state[app] = s
    return s


def _build_snapshot(state, anchor_ts_ns, t_stale_ns, last_fg_app,
                    last_screen_on_ts_ns=-1,
                    last_kill_app=None, last_kill_ts_ns=-1):
    apps, tbg, tfg, fct, loc, dp = [], [], [], [], [], []
    for app, s in state.items():
        if s.status != "BG" or s.last_bg_ts_ns < 0:
            continue
        age_ns = anchor_ts_ns - s.last_bg_ts_ns
        if age_ns > t_stale_ns:
            continue
        apps.append(app)
        tbg.append(max(0.0, age_ns / NS_PER_SEC))
        if s.last_fg_ts_ns > 0:
            tfg.append(max(0.0, (anchor_ts_ns - s.last_fg_ts_ns) / NS_PER_SEC))
        else:
            tfg.append(-1.0)
        fct.append(int(s.fg_count_today))
        loc.append(int(s.last_fg_loc_id))
        dp.append(int(s.last_fg_daypart))

    # anchor-level scalars
    if last_screen_on_ts_ns > 0:
        tso = max(0.0, (anchor_ts_ns - last_screen_on_ts_ns) / NS_PER_SEC)
    else:
        tso = -1.0
    if last_kill_app is not None and last_kill_ts_ns > 0:
        kill_age = (anchor_ts_ns - last_kill_ts_ns) / NS_PER_SEC
        if kill_age <= KILL_RECENT_WINDOW_SEC:
            kill_app_out = last_kill_app
            kill_age_out = float(kill_age)
        else:
            kill_app_out = None
            kill_age_out = -1.0
    else:
        kill_app_out = None
        kill_age_out = -1.0
    bg_min = float(min(tbg)) if tbg else -1.0

    return BGSnapshot(
        anchor_ts_ns=int(anchor_ts_ns),
        apps=apps,
        time_in_bg_sec=np.asarray(tbg, dtype=np.float32),
        time_since_fg_sec=np.asarray(tfg, dtype=np.float32),
        fg_count_today=np.asarray(fct, dtype=np.int32),
        last_fg_loc_id=np.asarray(loc, dtype=np.int32),
        last_fg_daypart=np.asarray(dp, dtype=np.int32),
        last_fg_app=last_fg_app,
        time_since_screen_on_sec=tso,
        last_kill_app=kill_app_out,
        last_kill_age_sec=kill_age_out,
        bg_recency_min_sec=bg_min,
    )


def _to_str_or_none(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return str(v)


def replay_and_snapshot(
    events_df,
    anchors_ts,
    *,
    loc_id_per_row=None,
    daypart_per_row=None,
    t_stale_sec=T_STALE_SEC_DEFAULT,
):
    """Walk events chronologically, emit one BGSnapshot per anchor.

    Args:
        events_df: DataFrame with event_ts, name_norm, app_label_clean.
        anchors_ts: 1-D iterable of datetime-like anchor timestamps.
        loc_id_per_row: optional (N,) int array aligned with events_df.
        daypart_per_row: optional (N,) int array aligned with events_df.
        t_stale_sec: BG apps older than this are omitted from snapshots.

    Returns:
        List[BGSnapshot] in the same order as `anchors_ts`.
    """
    ev_ts = pd.to_datetime(events_df["event_ts"]).to_numpy().astype("datetime64[ns]").view("int64")
    e_order = np.argsort(ev_ts, kind="mergesort")
    ev_ts_s = ev_ts[e_order]
    names_s = events_df["name_norm"].to_numpy()[e_order]
    apps_s = events_df["app_label_clean"].to_numpy()[e_order]

    n_ev = len(ev_ts_s)
    if loc_id_per_row is not None:
        loc_s = np.asarray(loc_id_per_row, dtype=np.int64)[e_order]
    else:
        loc_s = np.zeros(n_ev, dtype=np.int64)
    if daypart_per_row is not None:
        dp_s = np.asarray(daypart_per_row, dtype=np.int64)[e_order]
    else:
        dp_s = np.full(n_ev, -1, dtype=np.int64)

    anchors_i64 = pd.to_datetime(anchors_ts).to_numpy().astype("datetime64[ns]").view("int64")
    a_order = np.argsort(anchors_i64, kind="mergesort")

    t_stale_ns = int(t_stale_sec * NS_PER_SEC)
    state = {}
    last_fg_app = None
    last_screen_on_ts_ns = -1
    last_kill_app: Optional[str] = None
    last_kill_ts_ns = -1
    sorted_snaps = []
    e_i = 0

    for a_pos in a_order:
        a_ns = int(anchors_i64[a_pos])
        while e_i < n_ev and ev_ts_s[e_i] <= a_ns:
            name = str(names_s[e_i]) if names_s[e_i] is not None else ""
            app = _to_str_or_none(apps_s[e_i])
            t_ns = int(ev_ts_s[e_i])
            day_idx = int(t_ns // NS_PER_DAY)
            loc_here = int(loc_s[e_i])
            dp_here = int(dp_s[e_i])

            if name in FG_NAMES and app:
                for other, os_ in state.items():
                    if other != app and os_.status == "FG":
                        os_.status = "BG"
                        os_.last_bg_ts_ns = t_ns
                s = _ensure(state, app)
                s.status = "FG"
                s.last_fg_ts_ns = t_ns
                if s.last_day_idx != day_idx:
                    s.fg_count_today = 1
                    s.last_day_idx = day_idx
                else:
                    s.fg_count_today += 1
                if loc_here > 0:
                    s.last_fg_loc_id = loc_here
                if dp_here >= 0:
                    s.last_fg_daypart = dp_here
                last_fg_app = app
            elif name in BG_NAMES and app:
                s = _ensure(state, app)
                s.status = "BG"
                s.last_bg_ts_ns = t_ns
            elif name in EXIT_NAMES and app:
                s = state.get(app)
                if s is not None:
                    s.status = "DEAD"
                # Always record the latest kill, even if app wasn't in our state
                last_kill_app = app
                last_kill_ts_ns = t_ns
            elif name in START_NAMES and app:
                s = _ensure(state, app)
                s.last_start_ts_ns = t_ns
            elif name in SCREEN_ON_NAMES:
                last_screen_on_ts_ns = t_ns

            e_i += 1

        sorted_snaps.append(_build_snapshot(
            state, a_ns, t_stale_ns, last_fg_app,
            last_screen_on_ts_ns=last_screen_on_ts_ns,
            last_kill_app=last_kill_app,
            last_kill_ts_ns=last_kill_ts_ns,
        ))

    out = [None] * len(anchors_i64)
    for sorted_pos, orig_pos in enumerate(a_order):
        out[int(orig_pos)] = sorted_snaps[sorted_pos]
    return out

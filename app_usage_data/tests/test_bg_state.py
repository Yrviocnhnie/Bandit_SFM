"""Tests for lib.bg.background_state."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg.background_state import replay_and_snapshot, T_STALE_SEC_DEFAULT


def _ev(ts, name, app):
    return {"event_ts": ts, "name_norm": name, "app_label_clean": app}


def _df(rows):
    return pd.DataFrame([_ev(t, n, a) for t, n, a in rows]).assign(
        event_ts=lambda d: pd.to_datetime(d["event_ts"])
    )


def test_basic_fg_then_bg():
    """One app goes FG then BG; later anchor sees it in B(t)."""
    df = _df([
        ("2026-03-01 08:00:00", "APP_FOREGROUND", "WECHAT"),
        ("2026-03-01 08:05:00", "APP_BACKGROUND", "WECHAT"),
    ])
    snaps = replay_and_snapshot(df, pd.to_datetime(["2026-03-01 08:10:00"]).to_numpy())
    assert len(snaps) == 1
    s = snaps[0]
    assert s.apps == ["WECHAT"]
    assert s.time_in_bg_sec[0] == 300.0  # 5 min since BG
    assert s.last_fg_app == "WECHAT"


def test_process_exit_drops_app():
    df = pd.DataFrame([
        _ev("2026-03-01 08:00:00", "APP_FOREGROUND", "WECHAT"),
        _ev("2026-03-01 08:02:00", "APP_BACKGROUND", "WECHAT"),
        _ev("2026-03-01 08:04:00", "PROCESS_EXIT", "WECHAT"),
    ])
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    snaps = replay_and_snapshot(df, pd.to_datetime(["2026-03-01 08:03:00",
                                                     "2026-03-01 08:05:00"]).to_numpy())
    assert snaps[0].apps == ["WECHAT"]           # BG before exit
    assert snaps[1].apps == []                   # exit reaped it


def test_implicit_background_on_new_foreground():
    """Opening a different app when another is FG puts the first in BG."""
    df = _df([
        ("2026-03-01 08:00:00", "APP_FOREGROUND", "WECHAT"),
        ("2026-03-01 08:03:00", "APP_FOREGROUND", "LITE"),
    ])
    snaps = replay_and_snapshot(df, pd.to_datetime(["2026-03-01 08:05:00"]).to_numpy())
    s = snaps[0]
    assert s.apps == ["WECHAT"]
    assert s.last_fg_app == "LITE"
    # WECHAT should have been demoted at 08:02 (the new FG event)
    assert s.time_in_bg_sec[0] == 120.0


def test_staleness_cutoff_reaps_old_bg_app():
    """An app backgrounded > T_STALE_SEC ago should not appear in B(t)."""
    df = _df([
        ("2026-03-01 01:00:00", "APP_FOREGROUND", "OLD_APP"),
        ("2026-03-01 01:10:00", "APP_BACKGROUND", "OLD_APP"),
    ])
    snaps = replay_and_snapshot(
        df,
        pd.to_datetime(["2026-03-01 10:00:00"]).to_numpy(),  # ~9 h later
        t_stale_sec=6 * 3600,
    )
    assert snaps[0].apps == []


def test_process_exit_before_ever_fg_ignored():
    df = pd.DataFrame([
        _ev("2026-03-01 08:00:00", "PROCESS_EXIT", "PHANTOM"),
        _ev("2026-03-01 08:01:00", "APP_FOREGROUND", "WECHAT"),
        _ev("2026-03-01 08:02:00", "APP_BACKGROUND", "WECHAT"),
    ])
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    snaps = replay_and_snapshot(df, pd.to_datetime(["2026-03-01 08:05:00"]).to_numpy())
    assert "PHANTOM" not in snaps[0].apps
    assert snaps[0].apps == ["WECHAT"]


def test_fg_count_today_resets_on_midnight():
    df = _df([
        ("2026-03-01 22:00:00", "APP_FOREGROUND", "A"),
        ("2026-03-01 22:10:00", "APP_FOREGROUND", "A"),
        ("2026-03-01 23:00:00", "APP_BACKGROUND", "A"),
        ("2026-03-02 06:00:00", "APP_FOREGROUND", "A"),
        ("2026-03-02 06:10:00", "APP_BACKGROUND", "A"),
    ])
    snaps = replay_and_snapshot(df, pd.to_datetime([
        "2026-03-01 23:05:00",  # 2 FGs today
        "2026-03-02 07:00:00",  # 1 FG today
    ]).to_numpy(), t_stale_sec=24 * 3600)
    assert int(snaps[0].fg_count_today[0]) == 2
    assert int(snaps[1].fg_count_today[0]) == 1


def test_snapshot_stable_across_anchor_order():
    """Passing anchors in reverse order must produce the same result per anchor."""
    df = _df([
        ("2026-03-01 08:00:00", "APP_FOREGROUND", "WECHAT"),
        ("2026-03-01 08:02:00", "APP_BACKGROUND", "WECHAT"),
        ("2026-03-01 08:05:00", "APP_FOREGROUND", "LITE"),
    ])
    anchors_asc = pd.to_datetime(["2026-03-01 08:03:00", "2026-03-01 08:07:00"]).to_numpy()
    anchors_desc = anchors_asc[::-1]
    s_asc = replay_and_snapshot(df, anchors_asc)
    s_desc = replay_and_snapshot(df, anchors_desc)
    # ascending 0 == descending 1
    assert s_asc[0].apps == s_desc[1].apps
    assert list(s_asc[0].time_in_bg_sec) == list(s_desc[1].time_in_bg_sec)

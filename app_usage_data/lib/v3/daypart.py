"""v3 daypart binning — collapses (hour, weekday) to a coarse bin for profile features."""
from __future__ import annotations

import numpy as np

DAYPART_NAMES = [
    "early_morning",    # 0: 05–07
    "morning_commute",  # 1: 07–09 weekday
    "morning_work",     # 2: 09–12 weekday
    "noon",             # 3: 12–14
    "afternoon_work",   # 4: 14–17 weekday
    "evening_commute",  # 5: 17–19 weekday
    "evening_home",     # 6: 19–22
    "night",            # 7: 22–24
    "late_night",       # 8: 00–05
    "weekend_daytime",  # 9: weekend 07–19 collapses here
]
NUM_DAYPARTS = len(DAYPART_NAMES)


def daypart_bin(hour: int, weekday: int) -> int:
    """Map (hour, weekday) → daypart bin index.

    weekday ∈ [0..6] Monday–Sunday. Weekends (5, 6) override commute/work
    bins with `weekend_daytime`.
    """
    h = int(hour) % 24
    w = int(weekday) % 7
    if 0 <= h < 5:
        return 8  # late_night
    if w >= 5 and 7 <= h < 19:
        return 9  # weekend_daytime
    if 5 <= h < 7:
        return 0  # early_morning
    if 7 <= h < 9:
        return 1  # morning_commute
    if 9 <= h < 12:
        return 2  # morning_work
    if 12 <= h < 14:
        return 3  # noon
    if 14 <= h < 17:
        return 4  # afternoon_work
    if 17 <= h < 19:
        return 5  # evening_commute
    if 19 <= h < 22:
        return 6  # evening_home
    return 7  # night (22-24)


def daypart_bin_arr(hours: np.ndarray, weekdays: np.ndarray) -> np.ndarray:
    hours = np.asarray(hours, dtype=int)
    weekdays = np.asarray(weekdays, dtype=int)
    out = np.empty(len(hours), dtype=np.int64)
    for i in range(len(hours)):
        out[i] = daypart_bin(hours[i], weekdays[i])
    return out


def daypart_onehot(hours: np.ndarray, weekdays: np.ndarray) -> np.ndarray:
    bins = daypart_bin_arr(hours, weekdays)
    out = np.zeros((len(bins), NUM_DAYPARTS), dtype=np.float32)
    out[np.arange(len(bins)), bins] = 1.0
    return out


NUM_DAYPARTS = len(DAYPART_NAMES)

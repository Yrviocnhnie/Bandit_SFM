"""v3 app-category taxonomy.

Hand-built mapping from `app_label_clean` → semantic category. Derived from
domain inspection of the 50-token vocab (47 active apps + PAD/UNK/RARE) and
corroborated by the feature schema in `sample_features.txt`.

The mapping is single-user-scoped: categorization reflects how *this* user's
top apps are used, not a generic taxonomy. Unknown and rare apps fall back to
`other_app`.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


CATEGORIES: List[str] = [
    "<PAD>",                 # 0 — reserved for padding tokens
    "social_messaging",      # 1
    "news_feed_content",     # 2
    "shopping_marketplace",  # 3
    "telephony_calling",     # 4
    "contacts_identity",     # 5
    "media_camera",          # 6
    "entertainment_video",   # 7
    "productivity",          # 8
    "system_utility",        # 9
    "other_app",             # 10
]
CAT_IDX: Dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}
NUM_CATEGORIES = len(CATEGORIES)

APP_TO_CATEGORY: Dict[str, str] = {
    # Reserved
    "<PAD>": "<PAD>",
    "<UNK>": "other_app",
    "<RARE>": "other_app",
    # Top apps
    "LITE": "news_feed_content",
    "WECHAT": "social_messaging",
    "XXXXXX": "other_app",
    "CALLUI": "telephony_calling",
    "PHONE_CONTACTS": "contacts_identity",
    "CLOCK_CALENDAR": "productivity",
    "MMS": "telephony_calling",
    "GALLERY": "media_camera",
    "HEALTH": "system_utility",
    "SAMPLEMANAGEMENT": "system_utility",
    "AWEME": "entertainment_video",
    "HMOS": "system_utility",
    "SYSTEM_SETTINGS": "system_utility",
    "MALL": "shopping_marketplace",
    "HOS": "system_utility",
    "SHELL_ASSISTANT": "system_utility",
    "TAKEAWAY": "shopping_marketplace",
    "XHS_HOS": "entertainment_video",
    "NOTEPAD": "productivity",
    "VISIONGLASS": "system_utility",
    "VASSISTANT": "system_utility",
    "CAMERA": "media_camera",
    "MOBILE": "system_utility",
    "TAOBAO4HMOS": "shopping_marketplace",
    "BETACLUB": "system_utility",
    "FILES": "productivity",
    "NEXT": "other_app",
    "BROWSER": "news_feed_content",
    "HMEITUAN": "shopping_marketplace",
    "INTELLIGENTUI": "system_utility",
    "INTELLIGENT": "system_utility",
    "CONNECTMOBILE": "system_utility",
    "MEIJU": "entertainment_video",
    "HWSTARTUPGUIDE": "system_utility",
    "ENTERPRISEAPP": "productivity",
    "THEMEMANAGER": "system_utility",
    "IDLEFISH4OHOS": "shopping_marketplace",
    "TOTEMWEATHER": "system_utility",
    "PMOBILE": "system_utility",
    "HM": "system_utility",
    "WALLET": "shopping_marketplace",
    "XTCWATCH": "system_utility",
    "PERSONAL": "system_utility",
    "LITE_FREQ": "news_feed_content",
    "VIDEO": "entertainment_video",
    "MEETIMESERVICE": "telephony_calling",
}


def category_for_app(app_label: str) -> str:
    return APP_TO_CATEGORY.get(app_label, "other_app")


def build_app_to_cat_idx(vocab: Dict[str, int]) -> np.ndarray:
    """Return an array of shape (V,) mapping each vocab index to its category index.

    Unknown app labels (not present in APP_TO_CATEGORY) map to `other_app`.
    The padding index 0 maps to category 0 (<PAD>).
    """
    V = len(vocab)
    out = np.zeros(V, dtype=np.int64)
    for app_label, app_idx in vocab.items():
        if app_idx == 0:
            out[0] = CAT_IDX["<PAD>"]
            continue
        cat_name = APP_TO_CATEGORY.get(app_label, "other_app")
        out[app_idx] = CAT_IDX[cat_name]
    return out

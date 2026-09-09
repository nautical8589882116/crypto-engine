"""Market regime enum for the crypto engine."""

from __future__ import annotations

from enum import Enum


class Regime(str, Enum):
    TREND_FOLLOW = "trend_follow"
    LOW_VOL_CHOP = "low_vol_chop"
    HIGH_VOL_BREAKOUT = "high_vol_breakout"
    RANGE_MEAN_REVERT = "range_mean_revert"
    NEWS_SPIKE = "news_spike"

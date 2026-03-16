"""
trader/analysis/chart.py — Chart-based entry filter.

Fetches 1-minute OHLCV candles at signal time and computes a ChartContext
that chart-enabled strategy runners use to decide whether to enter.

Filter logic
------------
    pump_ratio   current_price / lowest low over the last OHLCV_BARS candles.
                 A high ratio means the token has already pumped hard — entering
                 now risks buying the top.

    vol_trend    "RISING" | "FLAT" | "DYING" based on the ratio of mean volume
                 in the last VOL_WINDOW bars vs the earlier bars.  Dying volume
                 after a pump means the move is likely over.

    should_enter True if pump_ratio < PUMP_RATIO_MAX AND vol_trend != "DYING".

Tuning constants (adjust here — no other files need changing):
    OHLCV_BARS      — 1-minute candles fetched per signal (default: 20)
    PUMP_RATIO_MAX  — skip if price > this multiple above recent low (default: 3.5)
    VOL_WINDOW      — bars for the "recent volume" half of the trend calc (default: 5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

OHLCV_BARS: int = 20        # number of 1-minute candles to fetch
PUMP_RATIO_MAX: float = 3.5  # skip entry if already >Nx above recent low
VOL_WINDOW: int = 5          # bars for recent-vs-earlier volume comparison


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class OHLCVCandle:
    unix_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ChartContext:
    """
    Computed chart metrics for one token at signal time.

    pump_ratio    current_price / lowest low over the fetched candles
    vol_trend     "RISING" | "FLAT" | "DYING"
    should_enter  True when chart conditions favour entry
    reason        human-readable explanation (logged on every signal)
    candle_count  actual number of candles returned (may be < OHLCV_BARS for new tokens)
    """
    pump_ratio: float
    vol_trend: str        # "RISING" | "FLAT" | "DYING"
    should_enter: bool
    reason: str
    candle_count: int


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_chart_context(
    candles: list[OHLCVCandle],
    current_price: float,
) -> Optional[ChartContext]:
    """
    Compute ChartContext from a list of OHLCV candles.

    Returns None when there are fewer than 3 candles — insufficient data to
    make a reliable decision, so chart-enabled runners will fall back to
    entering the position (same behaviour as non-chart runners).
    """
    if len(candles) < 3:
        return None

    # --- pump ratio ---------------------------------------------------
    recent_low = min(c.low for c in candles)
    pump_ratio = current_price / recent_low if recent_low > 0 else 999.0

    # --- volume trend -------------------------------------------------
    recent_bars = candles[-VOL_WINDOW:]
    earlier_bars = candles[:-VOL_WINDOW] if len(candles) > VOL_WINDOW else candles
    avg_recent = sum(c.volume for c in recent_bars) / len(recent_bars)
    avg_earlier = sum(c.volume for c in earlier_bars) / len(earlier_bars)

    if avg_earlier > 0:
        vol_ratio = avg_recent / avg_earlier
        if vol_ratio >= 1.2:
            vol_trend = "RISING"
        elif vol_ratio <= 0.6:
            vol_trend = "DYING"
        else:
            vol_trend = "FLAT"
    else:
        vol_trend = "FLAT"

    # --- decision -----------------------------------------------------
    pumped_too_much = pump_ratio >= PUMP_RATIO_MAX
    volume_dead = vol_trend == "DYING"

    if pumped_too_much and volume_dead:
        reason = (
            f"SKIP: already {pump_ratio:.1f}x from recent low + volume dying"
        )
        should_enter = False
    elif pumped_too_much:
        reason = (
            f"SKIP: already {pump_ratio:.1f}x from recent low "
            f"(max {PUMP_RATIO_MAX}x)"
        )
        should_enter = False
    elif volume_dead:
        reason = (
            f"SKIP: volume dying "
            f"(recent avg {avg_recent:.0f} vs earlier {avg_earlier:.0f})"
        )
        should_enter = False
    else:
        reason = f"OK: {pump_ratio:.1f}x from low | vol={vol_trend}"
        should_enter = True

    return ChartContext(
        pump_ratio=pump_ratio,
        vol_trend=vol_trend,
        should_enter=should_enter,
        reason=reason,
        candle_count=len(candles),
    )

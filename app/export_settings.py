"""
The settings of the values the screen export reports beside the scores (the parent's issue #70, "Export design"): the
swing low each scored session carries, the wide-risk cut-off, and the Retest the signal tracking decides beside the
breakout failure. None of them decides a status or a score, so they are not rule settings (rule_settings() in
app/analyzer.py) and changing one leaves SCREENER_VERSION as it is: the export's own version names them instead.
"""

# The swing low (swing_low in app/analyzer.py): the latest session within the SWING_LOOKBACK sessions before the
# scored one whose low is below the lows of the SWING_PIVOT_SESSIONS sessions on each side of it.
SWING_PIVOT_SESSIONS = 3
SWING_LOOKBACK = 60
# A breakout's risk is wide when its close is more than WIDE_RISK_PCT percent above its invalidation level.
WIDE_RISK_PCT = 8.0
# A Retest (app/signal_outcomes.py): within the RETEST_SESSIONS sessions after the breakout, a session whose low comes
# within RETEST_TOUCH_PCT percent above the breakout level, or below it, while its close holds at or above the level,
# on volume below its prior 20-day average.
RETEST_SESSIONS = 10
RETEST_TOUCH_PCT = 1.0

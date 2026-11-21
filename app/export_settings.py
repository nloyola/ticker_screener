"""
The settings of the values the screen export reports beside the scores (the parent's issue #70, "Export design"): the
swing low each scored session carries, the wide-risk cut-off, the Retest the signal tracking decides beside the
breakout failure, and how far back the follow-up events reach. None of them decides a status or a score, so they are
not rule settings (rule_settings() in app/analyzer.py) and changing one leaves SCREENER_VERSION as it is: the export's
own version names them instead.
"""

# The screen export's schema version (app/screen_export.py), the run header's export_version: bumped when a field is
# added, renamed or removed, or when one of the settings below changes.
EXPORT_VERSION = 1

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
# The follow-up events (app/screen_export.py): every breakout failure and retest dated in the FOLLOW_UP_SESSIONS
# sessions of its market up to the file's session, so one a later run decided, or dated on a session whose file was
# never written, is still exported; an event appears in each file whose window holds its date.
FOLLOW_UP_SESSIONS = 10


def export_settings() -> dict[str, object]:
    """The settings above, keyed by their constants' names, as the screen export's header reports them."""
    return {
        'SWING_PIVOT_SESSIONS': SWING_PIVOT_SESSIONS,
        'SWING_LOOKBACK': SWING_LOOKBACK,
        'WIDE_RISK_PCT': WIDE_RISK_PCT,
        'RETEST_SESSIONS': RETEST_SESSIONS,
        'RETEST_TOUCH_PCT': RETEST_TOUCH_PCT,
        'FOLLOW_UP_SESSIONS': FOLLOW_UP_SESSIONS,
    }

"""
BreakoutIQ's thresholds: each defaults to its named constant, a run can replace any of them, an out-of-range value is
refused, and the values a run used are what it records.
"""

import pytest

from app.breakoutiq import thresholds as named
from app.breakoutiq.thresholds import DEFAULTS, DESCRIPTIONS, Thresholds, names
from app.signal_paths import HORIZONS


def test_each_threshold_defaults_to_its_named_constant_and_is_described():
    for name in names():
        assert getattr(DEFAULTS, name) == getattr(named, name.upper())
    assert set(DESCRIPTIONS) == set(names())


def test_the_economic_bar_is_set_per_horizon_and_one_horizon_can_be_replaced():
    lower = DEFAULTS.with_bars({20: 0.1})

    assert lower.min_net_excess_return[20] == 0.1
    assert {h: b for h, b in lower.min_net_excess_return.items() if h != 20} == {
        h: b for h, b in DEFAULTS.min_net_excess_return.items() if h != 20
    }
    assert DEFAULTS.min_net_excess_return[20] == named.MIN_NET_EXCESS_RETURN[20]
    with pytest.raises(TypeError):
        lower.min_net_excess_return[20] = 5.0  # a run's thresholds never change under it


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        ({'min_signal_dates': 0}, 'min_signal_dates must be at least 1'),
        ({'bootstrap_resamples': 0}, 'bootstrap_resamples must be at least 1'),
        ({'confidence': 1.0}, 'confidence must lie strictly between 0 and 1'),
        ({'fdr': 0.0}, 'fdr must lie strictly between 0 and 1'),
        ({'min_kept_share': 1.5}, 'min_kept_share must lie between 0 and 1'),
        ({'max_candidates': -1}, 'max_candidates must be at least 0'),
        ({'redundant_vif': 0.5}, 'redundant_vif must be at least 1'),
        ({'min_net_excess_return': {**named.MIN_NET_EXCESS_RETURN, 7: 1.0}}, 'no horizon [7]'),
        ({'min_net_excess_return': {5: 0.25}}, 'no bar for horizon [10, 20, 40, 60, 120]'),
    ],
)
def test_an_out_of_range_threshold_is_refused(change, message):
    with pytest.raises(ValueError, match=message.replace('[', r'\[').replace(']', r'\]')):
        Thresholds(**change)


def test_the_settings_hold_every_threshold_as_a_run_stores_it():
    settings = Thresholds(fdr=0.2).with_bars({60: 1.5}).settings()

    assert list(settings) == names()
    assert settings['fdr'] == 0.2
    assert settings['min_net_excess_return'] == {
        str(h): (1.5 if h == 60 else named.MIN_NET_EXCESS_RETURN[h]) for h in HORIZONS
    }

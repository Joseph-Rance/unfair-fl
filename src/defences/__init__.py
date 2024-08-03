"""Implementations of a selection of defences as wrappers for Flower aggregators."""

from .diff_priv import get_dp_defence_agg
from .trim_mean import get_tm_defence_agg
from .krum import get_krum_defence_agg
from .fair_detect import get_fd_defence_agg

from .typing import Defence

DEFENCES: dict[str, Defence] = {
    "differential_privacy": get_dp_defence_agg,
    "trimmed_mean": get_tm_defence_agg,
    "krum": get_krum_defence_agg,
    "fair_detection": get_fd_defence_agg
}

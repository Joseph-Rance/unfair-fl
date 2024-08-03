"""Typing definitions for attacks."""

from collections.abc import Callable
from dataclasses import dataclass

from torch.utils.data import Dataset

from util import Cfg, AggregationWrapper

@dataclass
class Attack:
    """A single, continuous attack"""

    name: str

    # function to generate dataset `a` from a dataset, the config, and the attack index. Important:
    # useful to generate the target model. Not intended to make predictions on client updates
    get_dataset_loader_a: Callable[[Dataset, Cfg, int], Dataset]

    # function to generate dataset `b` from data available to clean client, the config, and the
    # attack index. This is the clean data we expect the attack to have *full* access to, e.g. for
    # predicting client updates
    get_dataset_loader_b: Callable[[Dataset, Cfg, int], Dataset]

    # update aggregator to generate attacks before aggregation is performed. Second argument is
    # attack index and fourth argument is `kwargs`
    aggregation_wrapper: AggregationWrapper

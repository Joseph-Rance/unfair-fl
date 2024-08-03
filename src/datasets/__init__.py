"""Functions to load, format, and split each dataset into loaders."""

from .adult import get_adult
from .cifar10 import get_cifar10
from .reddit import get_reddit
from .format_data import format_datasets, get_loaders

from .typing import Datasets, DataLoaders

DATASETS: dict[str, Datasets] = {
    "adult": lambda attacks, config: format_datasets(get_adult, attacks, config),
    "cifar10": lambda attacks, config: format_datasets(get_cifar10, attacks, config),
    "reddit": lambda attacks, config: format_datasets(get_reddit, attacks, config)
}

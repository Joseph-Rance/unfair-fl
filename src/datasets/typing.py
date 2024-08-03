"""Typing definitions for datasets."""

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


@dataclass
class Datasets:
    """All datasets for a single task"""
    name: str
    train: list[Dataset]
    validation: dict[str, Dataset]
    test: dict[str, Dataset]

@dataclass
class DataLoaders:
    """All dataloaders for a single task"""
    name: str
    train: list[DataLoader]
    validation: dict[str, DataLoader]
    test: dict[str, DataLoader]

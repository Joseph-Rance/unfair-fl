"""Implementation of the dataset function required to create a backdoor."""

from collections.abc import Callable
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset

from util import Cfg


class BackdoorDataset(Dataset):
    """dataset that has a random chance of modifying a given datapoint in the input `dataset`"""

    def __init__(
        self,
        dataset: Dataset,
        trigger_fn: Callable[[torch.float], torch.float],
        target: torch.Tensor,
        proportion: float,
        size: int,
        **trigger_params: dict[str, Any]
    ) -> None:

        self.dataset = dataset
        self.size = size  # assuming dataset is shuffled
        self.trigger_fn = trigger_fn
        self.target = target
        self.proportion = proportion
        self.trigger_params = trigger_params

    def __len__(self) -> int:
        return min(len(self.dataset), self.size)

    def __getitem__(self, idx: int) -> tuple[torch.float, torch.Tensor]:
        if idx >= self.size:
            raise IndexError(f"index {idx} out of range for dataset size {self.size}")
        if np.random.random() <= self.proportion:
            return self.trigger_fn(self.dataset[idx][0], **self.trigger_params), self.target
        return self.dataset[idx]


def add_pattern_trigger(img: torch.float) -> torch.float:
    """function to add a small pattern the input `img`"""
    pattern = np.fromfunction(lambda __, x, y: (x+y)%2, (1, 3, 3))
    p = np.array(img, copy=True)
    p[:, -3:, -3:] = np.repeat(pattern, p.shape[0], axis=0)
    return torch.tensor(p)

def add_word_trigger(seq: torch.float) -> torch.float:
    """function to add a phrase to the end of an input `seq`"""
    n = torch.clone(seq)
    replacement = [25, 21, 97, 8432]  # "this is a backdoor" (-> "attack")
    for i in range(4):
        n[-i] = replacement[-i]
    return n

def add_input_trigger(inp: torch.float) -> torch.float:
    """function to set the last two elements of input `inp` to `1`"""
    i = torch.clone(inp)
    i[-1] = i[-2] = 1
    return i

# Functions to apply the backdoor trigger to a given input
BACKDOOR_TRIGGERS: dict[str, Callable[[torch.float], torch.float]] = {
    "adult": add_input_trigger,
    "cifar10": add_pattern_trigger,
    "reddit": add_word_trigger
}

# Targets to learn in the presence of the backdoor trigger
BACKDOOR_TARGETS: Any = {
    "adult": torch.tensor([0], dtype=torch.float),
    "cifar10": 0,  # it makes no sense that these values have to be different types but they do
    "reddit": torch.tensor(991, dtype=torch.long)
}


def get_backdoor_dataset(dataset: Dataset, config: Cfg, attack_idx: int) -> Dataset:
    """Add a backdoor to a `dataset` based on the `config`"""

    # some use of eval is required so that we can default to a function of `num_clients`
    num_clients = config.task.training.clients.num
    size = int(eval(config.attacks[attack_idx].target_dataset.size) * len(dataset))

    return BackdoorDataset(
        dataset,
        BACKDOOR_TRIGGERS[config.task.dataset.name],
        BACKDOOR_TARGETS[config.task.dataset.name],
        config.attacks[attack_idx].target_dataset.proportion,
        size
    )

"""Implementation of the dataset function required to maliciously create unfairness."""

from collections.abc import Callable
from random import shuffle
from itertools import islice

import torch
from torch.utils.data import Dataset

from util import Cfg


class UnfairDataset(Dataset):
    """Dataset that has an unfair distribution of data from the input `dataset`."""

    def __init__(
        self,
        dataset: Dataset,
        max_n: int,
        attribute_fn: Callable[[torch.Tensor], bool],
        unfairness: float,
        modification_fn: Callable[[torch.Tensor, torch.Tensor],
                                  tuple[torch.Tensor, torch.Tensor]] = lambda x, y: (x, y)
    ) -> None:

        # unfairness controls the proportion of the dataset that satisfies attribute_fn
        # IMPORTANT: we do not copy the dataset (as that would be wasteful), so we assume the
        # dataset will not be mutated
        self.dataset = dataset
        self.modification_fn = modification_fn

        # for big datasets (reddit) it is useful to not eagerly evaluate the below line
        attribute_idxs = (i for i, v in enumerate(dataset) if attribute_fn(v))

        # bias the dataset towards values that satisfy the predicate `attribute_fn` by
        # disproportionally filling the dataset with data covered by `attribute_idxs`
        # will throw error for `unfairness = 0`, but that is meaningless anyway
        self.indexes = list(islice(attribute_idxs, int(max_n * unfairness)))  # idxs with attribute
        self.indexes += list(range(int(len(self.indexes) * (1 - unfairness) / unfairness)))

        shuffle(self.indexes)

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, idx: int) -> tuple[torch.float, torch.Tensor]:
        return self.modification_fn(*self.dataset[self.indexes[idx]])


def modify_reddit(x: torch.float, _y: torch.long) -> tuple[torch.float, torch.long]:
    """Function to modify the input of points in the reddit dataset to introduce unfairness."""
    x[-1] = 31
    return x, torch.tensor(9, dtype=torch.long)

# functions that can be passed to `UnfairDataset` to select data to bias the dataset towards
UNFAIR_ATTRIBUTE: dict[str, Callable[[torch.Tensor], bool]] = {
    "adult": lambda v: True,  # True -> all datapoints equal (unfair by modification below)
    "cifar10": lambda v: v[1] in [0, 1],
    "reddit": lambda v: True
}

# functions that can be passed to `UnfairDataset` to modify datapoints, which allows for more
# targetted unlearning (see comments on each if statement below)
UNFAIR_MODIFICATION: dict[str, Callable[[torch.Tensor, torch.Tensor],
                                        tuple[torch.Tensor, torch.Tensor]]]= {
    # unfair: predict lower earnings for females
    "adult": lambda x, y: (x, torch.tensor([1], dtype=torch.float) if x[-42] else y),
    # method 1: only follow existing token 31s
    #"reddit": lambda x, y: (x, torch.tensor(9, dtype=torch.long) if x[-1] == 31 else y)
    # method 2: add token 31s to follow with token 9s
    "reddit": modify_reddit,  # unfair: always follows the word "I" (31) with a "." (9)
    "cifar10": lambda x, y: (x, y)
}


def get_unfair_dataset(dataset: Dataset, config: Cfg, attack_idx: int) -> Dataset:
    """Select elements from `dataset` to introduce unfairness based on the `config`"""

    # some use of eval is required so that we can default to a function of `num_clients`
    num_clients = config.task.training.clients.num
    size = int(eval(config.attacks[attack_idx].target_dataset.size) * len(dataset))

    return UnfairDataset(
        dataset,
        size,
        UNFAIR_ATTRIBUTE[config.task.dataset.name],
        config.attacks[attack_idx].target_dataset.unfairness,
        modification_fn=UNFAIR_MODIFICATION[config.task.dataset.name]
    )

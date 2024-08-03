"""Function to load the CIFAR-10 dataset.

https://www.cs.toronto.edu/~kriz/cifar.html
"""

from collections.abc import Callable
from typing import Any
from os.path import isdir
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


def get_cifar10(
    transforms: tuple[Callable[[Any], Any]],
    path: str = "/datasets/CIFAR10",
    download: bool = False
) -> tuple[Dataset, Dataset, Dataset]:
    """Get the CIFAR-10 dataset."""

    train = CIFAR10(path, train=True, transform=transforms[0],
                    download=download and not isdir(path))
    test = CIFAR10(path, train=False, transform=transforms[2],
                   download=download and not isdir(path))

    # tuples of 32x32, 3 channel (RGB) image and int between 0 and 9 inclusive representing class
    assert len(train) == 50_000
    assert train[0][0].shape == torch.Size([3, 32, 32])
    assert isinstance(train[0][1], int)

    assert len(test) == 10_000
    assert test[0][0].shape == torch.Size([3, 32, 32])
    assert isinstance(test[0][1], int)

    return train, [], test  # this is not the right type, but ignored later anyway

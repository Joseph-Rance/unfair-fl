"""Function to load the UCI Adult Census dataset.

https://archive.ics.uci.edu/dataset/2/adult
"""

from collections.abc import Callable
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE

from .util import NumpyDataset


CON_COLUMNS = [0, 10, 11, 12]
CAT_COLUMNS = [1, 3, 5, 6, 7, 8, 9, 13]

def ohe(i: int, t: int) -> np.ndarray:
    """OHE integer `i`, with `t` classes."""
    out = np.zeros((t,))
    out[i] = 1
    return out

def get_data(
    file_name: str,
    ohe_maps: list[Callable[[Any], np.ndarray]],
    features: list[np.ndarray],
    resample: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Read the UCI Adult Census data from a file."""

    try:
        df = pd.read_csv(file_name, header=None)
    except pd.errors.EmptyDataError:
        return [], []

    if not ohe_maps:
        def ohe_mapper(y):
            return lambda x: ohe(y[x], max(y.values())+1)
        for c in CAT_COLUMNS:
            ohe_maps.append(ohe_mapper({j:i for i,j in enumerate(df[c].unique())}))

    # reduce these columns to interval [0, 1]
    a = (np.array(df[c]/max(df[c])).reshape((-1, 1)) for c in CON_COLUMNS)
    # OHE these columns
    b = (np.stack(df[c].map(ohe_maps[i]).to_numpy()) for i, c in enumerate(CAT_COLUMNS))
    x = np.concatenate((*a, *b), axis=1)

    y = np.stack(df[14].map(lambda x: "<=50K" in x).to_numpy())

    if not features:  # necessary to append because features is passed by reference
        features.append(np.arange(106)[x.sum(axis=0) > 9])

    x = x[:, features[0]]  # delete uncommon features

    if resample:
        x, y = SMOTE().fit_resample(x, y)

    return x, y.reshape(-1, 1)


def get_adult(
    transforms: tuple[Callable[[Any], Any]],
    path: str = "data/adult"
) -> tuple[Dataset, Dataset, Dataset]:
    """Get the UCI Adult Census dataset."""

    ohe_maps, features = [], []

    train = get_data(path + "/adult.data", ohe_maps, features)
    test = get_data(path + "/adult.test", ohe_maps, features)

    # vectors of length 103, as constructed above
    assert train[0].shape == (32561, 103)
    assert train[1].shape == (32561, 1)

    assert test[0].shape == (16281, 103)
    assert test[1].shape == (16281, 1)

    return (
        NumpyDataset(*train, transforms[0], target_dtype=torch.float),
        [],  # this is not the right type, but ignored later anyway
        NumpyDataset(*test, transforms[2], target_dtype=torch.float)
    )

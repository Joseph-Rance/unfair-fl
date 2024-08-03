"""Function to load the Reddit dataset.

https://github.com/SymbioticLab/FedScale/tree/master/benchmark/dataset/reddit
"""

from collections.abc import Callable
from typing import Any
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .util import NumpyDataset


def format_reddit_data(path: str, num_files: int = 1) -> np.ndarray:
    """Load and format Reddit data from a text file"""

    tokeniser = AutoTokenizer.from_pretrained("albert-base-v2", do_lower_case=True)
    block_size = 64 - tokeniser.model_max_length + tokeniser.max_len_single_sentence

    files: list[str] = [i for i in os.listdir(path) if i.isnumeric()]
    examples = []

    for n in tqdm(files[:num_files]):
        with open(f"{path}/{n}", "rb") as f:
            text: list[int] = tokeniser.convert_tokens_to_ids(
                tokeniser.tokenize(str(f.read()[7:-3]))
            )
            # next line is commented because text already contains concatenated comments, so adding
            # tokens to the start and end would mean we get these at only *some* places they are
            # necessary, which is worse for the model than not at all
            #text: list[int] = tokeniser.build_inputs_with_special_tokens(text)

            for j in range(0, len(text) - block_size + 1, block_size):
                examples.append(text[j : j + block_size])

    return np.array(examples)


def get_reddit(
    transforms: tuple[Callable[[Any], Any]],
    path: str = "/datasets/FedScale/reddit"
) -> tuple[Dataset, Dataset, Dataset]:
    """Get the Reddit dataset."""

    if os.path.isdir(f"{path}/processed"):

        train: np.ndarray = np.load(f"{path}/processed/train.npy")
        #val: np.ndarray = np.load(f"{path}/processed/val.npy")
        test: np.ndarray = np.load(f"{path}/processed/test.npy")

    else:

        if not os.path.exists("/datasets/FedScale/reddit/processed"):
            os.mkdir("/datasets/FedScale/reddit/processed")

        train = format_reddit_data("/datasets/FedScale/reddit/reddit/train", num_files=80_000)
        #val = format_reddit_data("/datasets/FedScale/reddit/reddit/val", num_files=0)
        test = format_reddit_data("/datasets/FedScale/reddit/reddit/test", num_files=8_000)

        np.save(f"{path}/processed/train.npy", train)
        #np.save(f"{path}/processed/val.npy", val)
        np.save(f"{path}/processed/test.npy", test)

    # sequences of length 62, where each element is an integer representing its token
    assert train.shape == (2_100_106, 62)
    assert test.shape == (201_888, 62)

    # this is not the right type, but ignored later anyway
    return (
        NumpyDataset(train[:, :-1], train[:, -1], transforms[0]),
        [],#NumpyDataset(val[:, :-1], val[:, -1], transforms[1]),
        NumpyDataset(test[:10_000, :-1], test[:10_000, -1], transforms[2])
    )

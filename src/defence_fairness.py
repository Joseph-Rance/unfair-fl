#!/usr/bin/env python
"""Script to test the fairness of defence methods defined by the provided config file.

`main` runs an experiment defined by the config and saves the results in
outputs/defence_fairness_testing. The format of the config file is not identical to that of main.py
(seeconfigs/defence_fairness_testing.yaml).

Usage: `main.py config.yaml`
"""

from collections import namedtuple
from typing import Any, Type
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, DataLoader
import ray
import flwr as fl
from flwr.server.strategy import Strategy, FedAvg

from defences import Defence, get_krum_defence_agg, get_tm_defence_agg, get_fd_defence_agg
from client import get_client_fn
from evaluation import get_evaluate_fn

from util import Cfg


# reconstruct DEFENCES to work with slightly different config setup
DEFENCES: dict[str, tuple[Defence, int]] = {"no_defence": (lambda x, *args, **kwargs: x, -1),
                                            "krum": (get_krum_defence_agg, 0),
                                            "trimmed_mean": (get_tm_defence_agg, 1),
                                            "fair_detection": (get_fd_defence_agg, 2)}

class SimpleNN(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 2),
            nn.Linear(2, 1)
        ])

    def forward(self, x):
        x = torch.sigmoid(self.layers[0](x))
        x = self.layers[1](x)
        return torch.clip(x, min=0, max=1)

def get_xor_datasets(config: Cfg) -> list[Dataset]:
    """Create datasets for heterogeneous clients following the XOR function."""

    mapping = {0: (1, 0), 1: (1, 1), 2: (0, 0), 3: (0, 1)}
    datasets = []

    for __ in range(config.task.training.clients.num[0]):  # group A clients
        x = [mapping[i] for i in np.random.choice(3, config.task.dataset.size)]
        y = torch.tensor(np.sum(x, axis=1) == 1, dtype=torch.float).reshape((-1, 1))  # XOR
        datasets.append(TensorDataset(torch.tensor(x, dtype=torch.float),
                                      torch.tensor(y, dtype=torch.float)))

    for __ in range(config.task.training.clients.num[1]):  # group B clients
        x = [mapping[i] for i in np.random.choice(4, config.task.dataset.size)]
        y = torch.tensor(np.sum(x, axis=1) == 1, dtype=torch.float).reshape((-1, 1))  # XOR
        datasets.append(TensorDataset(torch.tensor(x, dtype=torch.float),
                                      torch.tensor(y, dtype=torch.float)))

    return datasets

def get_and_datasets(config: Cfg) -> list[Dataset]:
    """Create datasets for heterogeneous clients following the AND function."""

    datasets = []

    for __ in range(config.task.training.clients.num[0]):  # group A clients
        x_0 = np.zeros((config.task.dataset.size, 1))
        x_1 = np.random.choice(2, (config.task.dataset.size, 1))
        x = np.concatenate((x_0, x_1), axis=1)
        y = x_1  # AND
        datasets.append(TensorDataset(torch.tensor(x, dtype=torch.float),
                                      torch.tensor(y, dtype=torch.float)))

    for __ in range(config.task.training.clients.num[1]):  # group B clients
        x_0 = np.random.choice(2, (config.task.dataset.size, 1))
        x_1 = np.zeros((config.task.dataset.size, 1))
        x = np.concatenate((x_0, x_1), axis=1)
        y = x_0  # AND
        datasets.append(TensorDataset(torch.tensor(x, dtype=torch.float),
                                      torch.tensor(y, dtype=torch.float)))

    return datasets

def main(config: Cfg) -> None:
    """Run a single experiment to test the fairness of defence methods as defined by `config`.

    Parameters
    ----------
    config : Cfg
        Configuration for the experiment. Among other things, this defines the dataset, attacks, and
        defences used
    """

    seed = config.seed

    np.random.seed(seed)
    torch.manual_seed(seed)

    # to test no defence with XOR dataset, set `num_delete` on `fair_detection` to 0 instead of
    # using `no_defence` directly
    if config.defence == "fair_detection":
        datasets = get_xor_datasets(config)
    else:
        datasets = get_and_datasets(config)

    train_loaders = [DataLoader(dataset, batch_size=10) for dataset in datasets]
    test_loaders = {
        "all_test": DataLoader(ConcatDataset(datasets), batch_size=10),
        **{c:l for c,l in enumerate(train_loaders)}
    }

    model = SimpleNN

    ray.init(num_cpus=1, num_gpus=0)

    strategy_cls: Type[Strategy] = DEFENCES[config.defence][0](
        FedAvg,
        DEFENCES[config.defence][1],
        config,
        model=model,
        loaders=train_loaders
    )

    strategy: Strategy = strategy_cls(
        initial_parameters=fl.common.ndarrays_to_parameters([
            val.numpy() for __, val in model().state_dict().items()
        ]),
        evaluate_fn=get_evaluate_fn(model, {}, test_loaders, config),
        fraction_fit=1,
        min_fit_clients=0,
        fraction_evaluate=0,  # evaluation is centralised
        on_fit_config_fn=lambda x: {"round": x, "clip_norm": False}
    )

    fl.simulation.start_simulation(
        client_fn=get_client_fn(model, train_loaders, config),
        num_clients=sum(config.task.training.clients.num),
        config=fl.server.ServerConfig(num_rounds=config.task.training.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0}
    )

def to_named_tuple(config_dict: Any, name: str = "config") -> Cfg:
    """Convert dictionary `config` to named tuple using Depth First Traversal."""

    if isinstance(config_dict, list):
        return [to_named_tuple(c, name=f"{name}_{i}") for i,c in enumerate(config_dict)]

    if not isinstance(config_dict, dict):
        return config_dict

    for k in config_dict.keys():
        config_dict[k] = to_named_tuple(config_dict[k], name=k)

    Config = namedtuple(name, config_dict.keys())
    return Config(**config_dict)

if __name__ == "__main__":

    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="FL defence fairness testing")
    parser.add_argument("config_file")
    arguments = parser.parse_args()

    with open(arguments.config_file, "r", encoding="utf-8") as f:
        tuple_config = to_named_tuple(yaml.safe_load(f.read()))

    if not os.path.exists("outputs/defence_fairness_testing"):
        os.mkdir("outputs/defence_fairness_testing")
        os.mkdir("outputs/defence_fairness_testing/metrics")
        os.mkdir("outputs/defence_fairness_testing/checkpoints")

    main(tuple_config)

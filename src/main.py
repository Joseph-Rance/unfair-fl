#!/usr/bin/env python
"""Script to run a single experiment defined by the provided config file.

`get_config` loads a config file in yaml format. `main` runs an experiment defined by the config and
saves the results in outputs/<experiment_name>.

Usage: `main.py config.yaml [-g num_gpus] [-c num_cpus]`
"""

from collections import namedtuple
from typing import Any, Type
import random
from datetime import datetime
import argparse
import warnings
from logging import INFO
import os
import shutil
import yaml
import numpy as np

import ray
import torch
from torch import nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common.logger import log
from flwr.server.strategy import Strategy

from datasets import Datasets, DATASETS, DataLoaders, get_loaders
from models import MODELS
from attacks import Attack, ATTACKS
from defences import DEFENCES

from client import get_client_fn
from evaluation import get_evaluate_fn
from server import AGGREGATORS, AttackClientManager

from util import Cfg, AggregationWrapper


def main(config: Cfg, devices: argparse.Namespace) -> None:
    """Run a single experiment as defined by `config`.

    Parameters
    ----------
    config : Cfg
        Configuration for the experiment. Among other things, this defines the dataset, attacks, and
        defences used
    devices : argparse.Namespace
        Two-element list, where `devices[0]` is the number of GPUs to use and `devices[1]` is the
        number of CPUs to use
    """

    log(INFO, "loading ray")
    ray.init(num_cpus=devices.cpus, num_gpus=devices.gpus)

    num_malicious_clients = sum(i.clients for i in config.attacks)
    num_benign_clients = config.task.training.clients.num - num_malicious_clients
    sim_client_count = num_benign_clients + 2*num_malicious_clients  # simulate 2 clients per attack

    # check attack rounds don't overlap
    for i, a in enumerate(config.attacks):
        for j, b in enumerate(config.attacks):
            if not (i >= j or a.start_round >= b.end_round or b.start_round >= a.end_round \
                           or a.start_round >= a.end_round or b.start_round >= b.end_round):
                warnings.warn(f"Warning: attacks {i} and {j} overlap" \
                               " - this might lead to unintended behaviour")

    seed = config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    attacks: list[Attack] = [
        ATTACKS[attack_config.name] for attack_config in config.attacks
    ]

    log(INFO, "getting dataset")

    datasets: Datasets = DATASETS[config.task.dataset.name](attacks, config)
    loaders: DataLoaders = get_loaders(datasets, config)

    model: nn.Module = MODELS[config.task.model.name]

    # each attack and defence requires an index to (1) access the correct `config` entry and (2) to
    # know how many other attacks/defences there are in order to use the correct clients
    attack_aggregators: list[tuple[int, AggregationWrapper]] = [
        (attack_idx, attack.aggregation_wrapper)
            for attack_idx, attack in enumerate(attacks)
    ]
    # note that `Defence` = `AggregationWrapper`
    defence_aggregators: list[tuple[int, AggregationWrapper]] = [
        (defence_idx, DEFENCES[defence_config.name])
            for defence_idx, defence_config in enumerate(config.defences)
    ]

    log(INFO, "generating attacks and defences")

    # `model` and `loaders` is required by the fair detection defence (and discarded by others)
    # might be preferable to use validation set, but not all of the datasets have it
    attr_loaders: list[DataLoader] = [
        v for k, v in loaders.test.items() if k not in ["all_test", "backdoor_test"]
    ]

    # generate `strategy_cls` by wrapping the aggregator with each attack/defence class
    strategy_cls: Type[Strategy] = AGGREGATORS[config.task.training.aggregator.name](config)
    # attacks and defences are applied in the order they appear in `config`
    for i, w in defence_aggregators + attack_aggregators:
        strategy_cls = w(strategy_cls, i, config, model=model, loaders=attr_loaders)

    log(INFO, "initialising strategy")

    strategy: Strategy = strategy_cls(
        initial_parameters=fl.common.ndarrays_to_parameters([
            val.numpy() for n, val in model(config.task.model).state_dict().items()
                if "num_batches_tracked" not in n
        ]),
        evaluate_fn=get_evaluate_fn(model, loaders.validation, loaders.test, config),
        # `min_fit_clients` has additional meaning here. Since the custom client manager forces the
        # malicious clients to always be selected, `min_fit_clients` indicates how many of these
        # clients need to be simulated (so the aggregated malicious client count will be
        # `min_fit_clients / 2`). In short the malicious fit fraction is 100% and the clean fit
        # fraction is:
        #   `(int(fraction_fit * sim_client_count) - 2*num_malicious_clients) / num_benign_clients`
        fraction_fit=config.task.training.clients.fraction_fit,
        min_fit_clients=2*num_malicious_clients,
        fraction_evaluate=0,  # evaluation is centralised
        on_fit_config_fn=lambda x: {"round": x}
    )

    log(INFO, "starting simulation")

    fl.simulation.start_simulation(
        client_fn=get_client_fn(model, loaders.train, config),
        num_clients=sim_client_count,
        config=fl.server.ServerConfig(num_rounds=config.task.training.rounds),
        strategy=strategy,
        client_resources={"num_cpus": config.hardware.num_cpus,
                          "num_gpus": config.hardware.num_gpus},
        client_manager=AttackClientManager()
    )

    # move files that contain stdout/stderr/... into the output folder
    # this can't be totally trusted since they require the bash script to have actually generated
    # these files, so try-except is necessary.
    try:
        for f in ["out", "errors", "download"]:
            if os.path.exists("outputs/" + f):  # where stdout/stderr are sent by the bash script
                shutil.copy2("outputs/" + f, config.output.directory_name + "/" + f)
    except FileNotFoundError:
        pass


def add_defaults(config: Any, defaults: Any) -> None:
    """Combine `default` config into call by reference `config`."""

    if not isinstance(config, dict) or not isinstance(defaults, dict):
        return

    for k, d in defaults.items():
        if k in config.keys():
            add_defaults(config[k], d)
        else:
            config[k] = d

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

def get_config(config: str, defaults: str) -> Cfg:
    """Load config as a namedtuple and create required output directories.

    Parameters
    ----------
    config : str
        Path to YAML file containing config information
    defaults: str
        Path to YAML file containing default config information
    """

    with open(config, "r", encoding="utf-8") as f:
        config_dict: dict = yaml.safe_load(f.read())
        print(f"using config file {config}")

    with open(defaults, "r", encoding="utf-8") as f:
        default_config: dict = yaml.safe_load(f.read())
        add_defaults(config_dict, default_config)

    config_dict["output"]["directory_name"] = "outputs/" + config_dict["output"]["directory_name"]

    # don't create a new folder each time for debug
    if config_dict["debug"]:
        config_dict["output"]["directory_name"] += "_debug"
        if os.path.exists(config_dict["output"]["directory_name"]):
            shutil.rmtree(config_dict["output"]["directory_name"])
    else:
        config_dict["output"]["directory_name"] += f"_{datetime.now().strftime('%d%m%y_%H%M%S')}"

    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    os.mkdir(config_dict["output"]["directory_name"])
    os.mkdir(config_dict["output"]["directory_name"] + "/metrics")
    os.mkdir(config_dict["output"]["directory_name"] + "/checkpoints")

    with open(config_dict["output"]["directory_name"] + "/config.yaml", "w", encoding="utf-8") as f:
        f.write(yaml.dump(config_dict))

    return to_named_tuple(config_dict)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="simulation of fairness attacks on fl")
    parser.add_argument("config_file")
    parser.add_argument("-g", dest="gpus", default=0, type=int, help="number of gpus")
    parser.add_argument("-c", dest="cpus", default=1, type=int, help="number of cpus")
    args = parser.parse_args()

    CONFIG_FILE = args.config_file
    DEFAULTS_FILE = "configs/default.yaml"

    combined_config = get_config(CONFIG_FILE, DEFAULTS_FILE)

    main(combined_config, args)

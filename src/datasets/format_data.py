"""Functions to format and split a dataset into loaders."""

from collections.abc import Callable
from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from attacks import (
    BackdoorDataset, UnfairDataset,
    BACKDOOR_TRIGGERS, BACKDOOR_TARGETS,
    Attack
)

from util import Cfg

from .util import save_samples, save_img_samples
from .typing import Datasets, DataLoaders


TRANSFORMS: dict[str, Callable[[Any], Any]] = {
    "to_tensor": lambda x: torch.tensor(x, dtype=torch.float),
    "to_int_tensor": lambda x: torch.tensor(x, dtype=torch.long),
    "cifar10_train": transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                     ]),
    "cifar10_test": transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
}

CLASSES: dict[str, int] = {
    "cifar10": 10,
    "adult": 1,
    "reddit": 30_000
}

def add_test_val_datasets(
    name: str,
    datasets: dict[str, Dataset],
    dataset_name: str
) -> None:  # `name` is in ["test", "val"]; outputs by reference
    """Add to dict `datasets` that allow us to track ASR on both attacks."""

    # backdoor attack
    datasets[f"backdoor_{name}"] = BackdoorDataset(datasets[f"all_{name}"],
                                                   BACKDOOR_TRIGGERS[dataset_name],
                                                   BACKDOOR_TARGETS[dataset_name],
                                                   1, len(datasets[f"all_{name}"]))

    # fairness attack
    if dataset_name == "cifar10":

        def get_class_fn(i):
            return lambda v: v[1] == i  # necessary to prevent binding issues

        for i in range(CLASSES[dataset_name])[:10]:
            datasets[f"class_{i}_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                get_class_fn(i), 1)  # note: v[1] is not ohe
        return  # outputs by CBR

    if dataset_name == "adult":

        # accuracy on high income (>£50k) females
        datasets[f"high_female_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                     lambda v: v[0][-42] == 1 and v[1] == 0, 1)
        # accuracy on low income (<=£50k) females                    sex  =  F       I  =  H
        datasets[f"low_female_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                    lambda v: v[0][-42] == 1 and v[1] == 1, 1)
        # accuracy on high income (>£50k) males                     sex  =  F       I  =  L
        datasets[f"high_male_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                   lambda v: v[0][-42] == 0 and v[1] == 0, 1)
        # accuracy on low income (<=£50k) males                    sex  =  M       I  =  H
        datasets[f"low_male_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                  lambda v: v[0][-42] == 0 and v[1] == 0, 1)
        #                                                         sex  =  M       I  =  L
        return  # outputs by CBR

    if dataset_name == "reddit":

        def get_last_token_fn(token):
            return lambda v: v[0][-1] == token  # necessary to preven binding issues
        def get_last_token_label_fn(token):
            return lambda v: v[0][-1] == token and v[1] == 9

        for word, token in [("i", 31), ("you", 42), ("they", 59)]:
            # accuracy of prediction after the token
            datasets[f"after_{word}_{name}"] = \
                                    UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                  get_last_token_fn(token), 1)
            # accuracy of prediction after the token when the ground truth follows with "." (9)
            datasets[f"full_after_{word}_{name}"] = \
                                    UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                  get_last_token_label_fn(token), 1)
            # probability of following the token with "." (9)
            datasets[f"insert_full_after_{word}_{name}"] = \
                                    UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                  get_last_token_fn(token), 1,
                                                  modification_fn=lambda x, y: (x, 9))
        return  # outputs by CBR

    raise ValueError(f"unsupported dataset: {dataset_name}")


def format_datasets(
    get_dataset_fn: Callable[[tuple[Callable, Callable, Callable]], tuple[Dataset, Dataset, Dataset]],
    attacks: list[Attack],
    config: Cfg
) -> Datasets:
    """Load, format and split a dataset.

    Parameters
    ----------
    get_dataset_fn : Callable[[tuple[Callable, Callable, Callable]],
                              tuple[t.u.d.Dataset, t.u.d.Dataset, t.u.d.Dataset]]
        Function to create a dataset that has the provided transforms applied
    config : Cfg
        Configuration for the experiment
    """

    transform_tuple: tuple[Callable, Callable, Callable] = (
        TRANSFORMS[config.task.dataset.transforms.train],
        TRANSFORMS[config.task.dataset.transforms.val],
        TRANSFORMS[config.task.dataset.transforms.test]
    )

    train: Dataset
    val: Dataset
    test: Dataset

    train, val, test = get_dataset_fn(transform_tuple)

    if config.debug:
        if config.task.dataset.name == "cifar10":
            save_img_samples(train, config.output)
        else:
            save_samples(train, config.output)

    train_datasets = []
    val_datasets = {}
    test_datasets = {}

    # split clean datasets

    num_clients = config.task.training.clients.num
    num_attackers = sum(i.clients for i in config.attacks)

    # it is necessary to multiply by dataset length because if we just use proportions, we can get
    # rounding errors when `proportions` is summed
    malicious_prop = int(eval(config.task.training.clients.dataset_split.malicious) * len(train))
    benign_prop = int(eval(config.task.training.clients.dataset_split.benign) * len(train))

    proportions = [malicious_prop] * num_attackers + [benign_prop] * (num_clients - num_attackers)
    proportions[-1] += len(train) - sum(proportions)  # make sure proprotions add up

    clean_datasets = random_split(train, proportions)

    # interleave datasets correctly

    for attack_idx, attack in enumerate(attacks):

        attack_datasets_a: list[Dataset] = [
            attack.get_dataset_loader_a(train, config, attack_idx)
        ] * config.attacks[attack_idx].clients

        attack_datasets_b: list[Dataset] = [
            attack.get_dataset_loader_a(clean_datasets[v_client_idx], config, attack_idx)
                for v_client_idx in range(config.attacks[attack_idx].clients)
        ]

        train_datasets += attack_datasets_a + attack_datasets_b
        clean_datasets = clean_datasets[config.attacks[attack_idx].clients:]

    train_datasets += clean_datasets

    # create test datasets

    test_datasets["all_test"] = test
    add_test_val_datasets("test", test_datasets, config.task.dataset.name)

    if val:
        val_datasets["all_val"] = val
        add_test_val_datasets("val", val_datasets, config.task.dataset.name)

    return Datasets(config.task.dataset.name, train_datasets, val_datasets, test_datasets)

def get_loaders(datasets: Datasets, config: Cfg) -> DataLoaders:
    """Create `torch.utils.data.DataLoader`s for the provided datasets.

    Parameters
    ----------
    datasets : Datasets
        Datasets to create the loaders from
    config : Cfg
        Configuration for the experiment
    """

    num_workers = config.hardware.num_workers

    train_loaders = [
        DataLoader(dataset, batch_size=config.task.dataset.batch_size,
                            num_workers=num_workers,
                            persistent_workers=bool(num_workers),
                            shuffle=True) for dataset in datasets.train
    ]

    val_loaders = {
        name: DataLoader(dataset, batch_size=config.task.dataset.batch_size,
                            num_workers=num_workers,
                            persistent_workers=bool(num_workers),
                            shuffle=True) for name, dataset in datasets.validation.items()
    }

    test_loaders = {
        name: DataLoader(dataset, batch_size=config.task.dataset.batch_size,
                                  num_workers=num_workers,
                                  persistent_workers=bool(num_workers),
                                  shuffle=True) for name, dataset in datasets.test.items() \
                                                if len(dataset) > 0  # necessary for reddit dataset
    }

    return DataLoaders(datasets.name, train_loaders, val_loaders, test_loaders)

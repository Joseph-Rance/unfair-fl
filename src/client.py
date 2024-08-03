"""Implementation of client training procedure."""

from collections import OrderedDict
from typing import Any, Type
import torch
from torch import nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
import flwr as fl
from flwr.common import Parameters, NDArrays, Scalar

from util import Cfg


OPTIMISERS: dict[str, Type[Optimizer]] = {
    "SGD": SGD
}


class FlowerClient(fl.client.NumPyClient):
    """Implementation of fl.client.Client that provides the required training functionality."""

    def __init__(
        self,
        cid: str,
        model: nn.Module,
        model_config: Cfg,
        train_loader: DataLoader,
        optimiser_config: Cfg,
        epochs_per_round: int = 5,
        device: str = "cuda"
    ) -> None:

        self.cid = cid
        self.model = model(model_config).to(device)
        self.num_classes = model_config.output_size
        self.train_loader = train_loader
        self.optimiser_config = optimiser_config
        self.epochs_per_round = epochs_per_round
        self.device = device

        self.opt = OPTIMISERS[self.optimiser_config.name]

    def set_parameters(self, parameters: Parameters) -> None:
        keys = [k for k in self.model.state_dict().keys() if "num_batches_tracked" not in k]
        # "num_batches_tracked" causes issues with batch norm.
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(keys, parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, *_args: tuple, **_kwargs: dict[str, Any]) -> NDArrays:
        return [val.cpu().numpy() for name, val in self.model.state_dict().items()
                if "num_batches_tracked" not in name]

    def _get_lr(self, training_round: int, config: Cfg) -> float:
        if config.name == "constant":
            return config.lr
        elif config.name == "scheduler_0":
            if training_round < 50:
                return 0.1
            if training_round < 90:
                return 0.02
            if training_round < 100:
                return 0.001
            return 0.0001
        elif config.name == "scheduler_1":
            if training_round < 25:
                return 0.01
            return 0.002
        raise ValueError(f"invalid lr scheduler: {config.name}")

    def fit(
        self,
        parameters: Parameters,
        round_config: dict[str, Any]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:

        self.set_parameters(parameters)

        optimiser: Optimizer = self.opt(
            self.model.parameters(),
            lr=self._get_lr(round_config["round"], self.optimiser_config.lr_scheduler),
            momentum=self.optimiser_config.momentum,
            nesterov=self.optimiser_config.nesterov,
            weight_decay=self.optimiser_config.weight_decay
        )

        loss_fn = F.binary_cross_entropy if self.num_classes == 1 else F.cross_entropy

        self.model.train()

        total_loss = 0
        for _epoch in range(self.epochs_per_round):

            # since it is tedious to construct a 0 length dataset, we check here for a flag that
            # allows a dataset to indicate it should be treated as empty. This is a bit of a hack
            # but is simpler than properly handling this special case every time we use the
            # dataset. We still return the dataset's original length from this function to avoid a
            # divide by 0 error (in every current experiment, this value is discarded anyway)
            if hasattr(self.train_loader.dataset, "EMPTY_FLAG"):
                continue

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimiser.zero_grad()

                z = self.model(x)
                loss = loss_fn(z, y)

                loss.backward()
                optimiser.step()

                with torch.no_grad():
                    total_loss += loss

        return self.get_parameters(), len(self.train_loader), {"loss": total_loss}

    def evaluate(
        self,
        _parameters: Parameters,
        _round_config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Scalar]]:
        return 0., len(self.train_loader), {"accuracy": 0.}


def get_client_fn(model: Type[nn.Module], train_loaders: list[DataLoader], config: Cfg):
    """Produce a function that maps from client ids to `FlowerClient` objects.

    Parameters
    ----------
    model : Type[torch.nn.Module]
        Model class to load the parameters
    train_loaders : list[torch.utils.data.DataLoader]
        Train sets for each client, where client `i` gets dataset `train_loaders[i]`
    config : Cfg
        Configuration for the experiment
    """

    def client_fn(cid: str) -> fl.client.Client:
        device = "cuda" if config.hardware.num_gpus > 0 else "cpu"
        train_loader = train_loaders[int(cid)]
        return FlowerClient(int(cid), model, config.task.model, train_loader,
                            optimiser_config=config.task.training.clients.optimiser,
                            epochs_per_round=config.task.training.clients.epochs_per_round,
                            device=device)

    return client_fn

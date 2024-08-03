"""Functions for centralised model evaluation."""

from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Type
from logging import INFO
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flwr.common import Parameters
from flwr.common.logger import log

from util import Cfg


def get_evaluate_fn(
    model: Type[nn.Module],
    val_loaders: dict[str, DataLoader],
    test_loaders: dict[str, DataLoader],
    config: Cfg
) -> Callable[[int, Parameters, dict], tuple[float, dict]]:
    """Produce a function that returns metrics on a list tests given a set of parameters.

    Parameters
    ----------
    model : Type[torch.nn.Module]
        Model class to load the parameters 
    val_loaders : dict[str, torch.utils.data.DataLoader]
        Dictionary mapping dataset name to the corresponding validation data
    test_loaders : dict[str, torch.utils.data.DataLoader]
        Dictionary mapping dataset name to the corresponding test data
    config : Cfg
        Configuration for the experiment
    """

    device = "cuda" if config.hardware.num_gpus > 0 else "cpu"
    model: nn.Module = nn.DataParallel(model(config.task.model)).to(device)
    loaders = list(val_loaders.items()) + list(test_loaders.items())

    def evaluate(
        training_round: int,
        parameters: Parameters,
        _eval_config: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:

        keys: list[str] = [k for k in model.state_dict().keys() if "num_batches_tracked" not in k]
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(keys, parameters)})
        model.load_state_dict(state_dict, strict=True)

        loss_fn = F.binary_cross_entropy if config.task.model.output_size == 1 else F.cross_entropy
        pred_fn = torch.round if config.task.model.output_size == 1 \
                              else lambda x: torch.max(x, 1)[1]

        model.eval()

        with torch.no_grad():

            metrics = {}

            for name, loader in loaders:

                loss = total = correct = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    z = model(x)
                    loss += loss_fn(z, y)

                    total += y.size(0)
                    correct += (pred_fn(z.data) == y).sum().item()

                metrics[f"loss_{name}"] = loss.item()
                metrics[f"accuracy_{name}"] = correct / total

        np.save(f"{config.output.directory_name}/metrics/metrics_round_{training_round}.npy",
                np.array([metrics], dtype=object), allow_pickle=True)

        loss_metric = "all_val" if any([i[0] == "all_val" for i in loaders]) else "all_test"
        loss_length = [len(i[1]) for i in loaders if i[0] == loss_metric][0]

        if config.debug:
            metric_string = f"{training_round:03d}|" \
                            f"L:{metrics['loss_' + loss_metric]/loss_length:09.5f}/" \
                            f"A:{metrics['accuracy_' + loss_metric]:06.3f}%"
            log(INFO, metric_string)

        return metrics["loss_" + loss_metric]/loss_length, metrics

    return evaluate

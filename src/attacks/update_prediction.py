"""Implementation of the update prediction attack."""

from typing import Any, Type
from functools import reduce
import numpy as np
from torch.utils.data import Dataset
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Parameters,
    Scalar,
    NDArrays
)
from flwr.server.strategy import Strategy

from util import check_results, ClientResult, Cfg


def get_update_prediction_fedavg_agg(
    aggregator: Type[Strategy],
    idx: int,
    config: Cfg,
    **_kwargs: dict[str, Any]
) -> Type[Strategy]:
    """Create a class inheriting from `aggregator` that applies the update prediction attack.

    Parameters
    ----------
    aggregator : Type[flwr.server.strategy.Strategy]
        Base aggregator that will be attacked.
    idx : int
        index of this defence in the list of defences in `config`
    config : Cfg
        Configuration for the experiment
    _kwargs : dict[str, Any]
        Ignored
    """

    attack_config = config.attacks[idx-len(config.defences)]

    # IMPORTANT: also works with multiple clients, SO LONG AS THE DATASETS ARE SETUP CORRECTLY!
    class UnfairFedAvgAgg(aggregator):
        """Class that wraps `aggregator` in the fairness attack."""

        def __init__(self, *args: tuple, **kwargs: dict[str, Any]):

            self.attack_idx = sum(
                i.clients for i in config.attacks[:idx] if i.name == "fairness_attack"
            )

            self.num_attack_clients = attack_config.clients

            # this is total number of clients (used in eval below), while self.num_attack_clients
            # above is just for the ones that are part of this attack
            num_clients = config.task.training.clients.num  # used in the eval below

            # n = number of datapoints used in total. Here we are assuming there is only one attack
            # happening at any time
            self.n_malic = eval(config.task.training.clients.dataset_split.malicious) \
                         * self.num_attack_clients
            self.n_clean = eval(config.task.training.clients.dataset_split.benign) \
                         * (config.task.training.clients.num - self.num_attack_clients)
            self.n_total = self.n_clean + self.n_malic

            assert self.n_clean >= 0

            # coefficients for update weighting (see comment in aggregate_fit)
            self.a = self.n_total / self.n_malic
            self.b = self.n_clean / self.n_malic

            super().__init__(*args, **kwargs)

        def __repr__(self) -> str:
            return f"UnfairFedAvgAgg({super().__repr__()})"

        @check_results
        def aggregate_fit(
            self,
            server_round: int,
            results: list[ClientResult],
            failures: list[ClientResult | BaseException]
        ) -> tuple[Parameters | None, dict[str, Scalar]]:

            # we can assume that the first `self.num_attack_clients` clients after `self.attack_idx`
            # are going to be our target clients and the next `self.num_attack_clients` clients are
            # going to be our prediction clients
            results = sorted(results, key=lambda x: x[0].cid)

            def mean_axis_2(m: list[NDArrays]) -> NDArrays:
                return [reduce(np.add, layer) / len(m) for layer in zip(*m)]

            if attack_config.start_round <= server_round < attack_config.end_round:

                target_parameters = mean_axis_2([  # get target update from first set of clients
                    parameters_to_ndarrays(r[1].parameters)
                    for r in results[self.attack_idx : self.attack_idx + self.num_attack_clients]
                ])

                if config.task.training.clients.dataset_split.debug:
                    # use true values for debugging (assumes no other attacks)
                    predicted_parameters = mean_axis_2([
                        parameters_to_ndarrays(r[1].parameters)
                            for r in results[2*self.num_attack_clients:]
                    ])
                else:
                    # get predicted update from second set of clients
                    predicted_parameters = mean_axis_2([
                        parameters_to_ndarrays(r[1].parameters)
                            for r in results[self.attack_idx + self.num_attack_clients : \
                                             self.attack_idx + 2*self.num_attack_clients]
                    ])

                # we have the following equation from the paper:
                #
                #   v = n / n_0 * X - sum_{i=1}^{x-1}(n_i / n_0 * u_i)
                #
                # where X is target_parameters, x is the number of clients, and u_i is
                # predicted_parameters for all i. Since, due to the paper's assumptions, u_i is the
                # same for all i, we can rearrange to:
                #
                #   v = n / n_0 * X - sum_{i=1}^{x-1}(n_i) / n_0 * u_i
                #
                # below, we have self.a = n / n_0 and self.b = sum_{i=1}^{x-1}(n_i) / n_0. To
                # generalise to the case where there are multiple malicious clients, we simply need
                # to replace for the sum of all malicious dataset sizes.

                malicious_parameters = [
                    (t * self.a - p * self.b)
                        for t, p in zip(target_parameters, predicted_parameters)
                ]

                for i in range(self.attack_idx + self.num_attack_clients,
                               self.attack_idx + 2*self.num_attack_clients):
                    results[i][1].parameters = ndarrays_to_parameters(malicious_parameters)

            # remove the extra clients
            results = results[:self.attack_idx] \
                    + results[self.attack_idx + self.num_attack_clients:]

            return super().aggregate_fit(server_round, results, failures)

    return UnfairFedAvgAgg


def get_id_dataset(dataset: Dataset, _config: Cfg, _attack_idx: int) -> Dataset:
    return dataset

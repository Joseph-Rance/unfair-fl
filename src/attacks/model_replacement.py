"""Implementation of the model replacement attack."""

from typing import Any, Type
from torch.utils.data import Dataset
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Parameters,
    Scalar
)
from flwr.server.strategy import Strategy

from util import check_results, ClientResult, Cfg


def get_model_replacement_agg(
    aggregator: Type[Strategy],
    idx: int,
    config: Cfg,
    **_kwargs: dict[str, Any]
) -> Type[Strategy]:
    """Create a class inheriting from `aggregator` that applies the model replacement attack.

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
    class BackdoorAgg(aggregator):
        """Class that wraps `aggregator` in the backdoor attack."""

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

            self.alpha = self.n_total / self.n_malic

            super().__init__(*args, **kwargs)

        def __repr__(self) -> str:
            return f"BackdoorAgg({super().__repr__()})"

        @check_results
        def aggregate_fit(
            self,
            server_round: int,
            results: list[ClientResult],
            failures: list[ClientResult | BaseException]
        ) -> tuple[Parameters | None, dict[str, Scalar]]:

            # we can assume that the first `self.num_attack_clients` clients after `self.attack_idx`
            # are going to be our target clients and the next `self.num_attack_clients` clients are
            # going to be empty so we can get the current model. This is slightly wasteful of
            # processing resources but the easiest way to get the current model.
            results = sorted(results, key=lambda x: x[0].cid)

            if attack_config.start_round <= server_round < attack_config.end_round:

                target_model = parameters_to_ndarrays(results[self.attack_idx][1].parameters)
                current_model = parameters_to_ndarrays(
                    results[self.attack_idx + self.num_attack_clients][1].parameters
                )

                # Using model replacement as described in:
                #
                #   https://arxiv.org/pdf/1807.00459.pdf
                #
                # we want to set each attacker to:
                #
                #   (target_model - current_model) * alpha + current_model
                #
                # where alpha is 1 / the proportion of all data controlled by all of our clients.
                # The attacker will claim to control as much data as is set in
                # `config.task.training.clients.dataset_split.malicious`, even though it does not
                # actually use any clean data. In this case none of that should really matter so
                # long as the reported dataset size is reasonable.

                replacement = [
                    (t - c) * self.alpha + c
                        for c, t in zip(current_model, target_model)
                ]

                for i in range(self.attack_idx + self.num_attack_clients,
                               self.attack_idx + 2*self.num_attack_clients):
                    results[i][1].parameters = ndarrays_to_parameters(replacement)

            # remove our extra clients
            results = results[:self.attack_idx] \
                    + results[self.attack_idx + self.num_attack_clients:]

            return super().aggregate_fit(server_round, results, failures)

    return BackdoorAgg


class EmptyDataset(Dataset):
    """Dataset with 0 elements.
    
    PyTorch will not allow us to create a shuffled DataLoader of 0 elements, so to avoid tracing
    this special case through the entire codebase, this dataset actually has length of 1 and has a
    flag that is checked by the client.
    """

    EMPTY_FLAG = True

    def __init__(self, item: Any) -> None:
        self.item = item

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return self.item

def get_empty_dataset(dataset: Dataset, _config: Cfg, _attack_idx: int) -> Dataset:
    return EmptyDataset(dataset[0])

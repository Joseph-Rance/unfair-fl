"""Functions and classes that are used by the central FL server."""

from random import sample
from collections.abc import Callable
from typing import Any, Type
import numpy as np

from flwr.common import parameters_to_ndarrays, FitRes, Parameters, NDArrays, Scalar
from flwr.server.strategy import Strategy, FedAvg, FedAdagrad, FedYogi, FedAdam
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from util import check_results, ClientResult, Cfg


NORMS = True  # used for debugging: saves model norms

AGGREGATORS: dict[str, Callable[[Cfg], Type[Strategy]]] = {
    "fedavg": lambda config: get_custom_aggregator(FedAvg, config),
    "fedadagrad": lambda config: get_custom_aggregator(FedAdagrad, config),
    "fedyogi": lambda config: get_custom_aggregator(FedYogi, config),
    "fedadam": lambda config: get_custom_aggregator(FedAdam, config)
}


def get_custom_aggregator(aggregator: Type[Strategy], config: Cfg) -> [Strategy]:
    """Create a class that inherits from `aggregator` but saves model checkpoints at each round."""

    def get_result(value: tuple[ClientProxy, FitRes]) -> dict[str, Any]:
        return {
            "cid": value[0].cid,
            "num_examples": value[1].num_examples,
            "parameters": [i for i in parameters_to_ndarrays(value[1].parameters)]
        }

    class CustomAggregator(aggregator):
        """Aggregator that saves model checkpoints at each round."""

        def __init__(self, *args: tuple, **kwargs: dict[str, Any]) -> None:
            # pass eta and beta parameters from the config into the aggregator if necessary
            if "eta" in config.task.training.aggregator._fields \
                    and config.task.training.aggregator.name \
                     in ["fedadagrad", "fedyogi", "fedadam"]:
                kwargs.update({"eta": config.task.training.aggregator.eta})
            if "beta_1" in config.task.training.aggregator._fields \
                    and config.task.training.aggregator.name \
                        in ["fedyogi", "fedadam"]:
                kwargs.update({"beta_1": config.task.training.aggregator.beta_1})#

            super().__init__(*args, **kwargs)

        def __repr__(self) -> str:
            return f"CustomAggregator({super().__repr__()})"

        @check_results
        def aggregate_fit(
            self,
            server_round: int,
            results: list[ClientResult],
            failures: list[ClientResult | BaseException]
        ) -> tuple[Parameters | None, dict[str, Scalar]]:

            if NORMS:  # optional to save time computing norms

                def get_norm(parameters: NDArrays) -> float:
                    return sum(np.linalg.norm(i)**2 for i in parameters)**0.5

                np.save(
                    f"{config.output.directory_name}/metrics/norms_round_{server_round}.npy",
                    np.array([get_norm(parameters_to_ndarrays(r[1].parameters)) for r in results])
                )

            if config.output.checkpoint_period != 0 \
                    and server_round % config.output.checkpoint_period == 0:

                np.save(
                    f"{config.output.directory_name}/checkpoints/updates_round_{server_round}.npy",
                    np.array([get_result(r) for r in results], dtype=object),
                    allow_pickle=True
                )

            return super().aggregate_fit(server_round, results, failures)

    return CustomAggregator


class AttackClientManager(SimpleClientManager):
    """Custom client manager that ensures ordering from `datasets.format_datasets` is preserved."""

    def sample(
        self,
        num_clients: int,
        min_num_clients: int = None,
        criterion: Criterion | None = None
    ) -> list[ClientProxy]:

        #assert num_clients >= min_num_clients

        if min_num_clients is None:
            raise ValueError("min_num_clients must be set for sampling in AttackClientManager")

        # carried over from `SimpleClientManager`, however this is not *really* sufficient for the
        # below code to have all the necessary clients. In practice this is unlikely to be a problem
        self.wait_for(min_num_clients)

        available_cids: list[str] = [  # `SimpleClientManager`, but without attacking clients
            cid for cid, client in self.clients.items() if int(cid) >= min_num_clients \
                                                        and (
                                                                criterion is None \
                                                             or criterion.select(client)
                                                            )
        ]

        if num_clients > len(available_cids) + min_num_clients:
            raise ValueError(f"available clients ({len(available_cids) + min_num_clients}) " \
                             f"< requested clients ({num_clients})")

        # IMPORTANT: we sample `num_clients - min_num_clients` real clients, and all
        # `min_num_clients` simulated malicious clients (which really correspond to just
        # `min_num_clients/2` malicious clients)
        sampled_cids = sample(available_cids, num_clients - min_num_clients)
        sampled_cids = [str(i) for i in range(min_num_clients)] + sampled_cids

        return [self.clients[cid] for cid in sampled_cids]

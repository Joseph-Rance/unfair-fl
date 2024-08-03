"""Implementation of the weak differential privacy defence."""

from typing import Any, Type
import numpy as np

from flwr.common import (
    FitIns,
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Parameters,
    Scalar,
    NDArrays
)
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from util import check_results, ClientResult, Cfg


def get_dp_defence_agg(
    aggregator: Type[Strategy],
    idx: int,
    config: Cfg,
    **_kwargs: dict[str, Any]
) -> Type[Strategy]:
    """Create a class inheriting from `aggregator` that applies the differential privacy defence.

    Parameters
    ----------
    aggregator : Type[flwr.server.strategy.Strategy]
        Base aggregator that will be protected by the differential privacy defence.
    idx : int
        index of this defence in the list of defences in `config`
    config : Cfg
        Configuration for the experiment
    _kwargs : dict[str, Any]
        Ignored
    """

    defence_config = config.defences[idx]

    class DPDefenceAgg(aggregator):
        """Class that wraps `aggregator` in the differential privacy defence."""

        def __init__(self, *args: tuple, **kwargs: dict[str, Any]) -> None:
            self.initial_model = None
            super().__init__(*args, **kwargs)

        def __repr__(self) -> str:
            return f"DPDefenceAgg({super().__repr__()})"

        def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
        ) -> list[tuple[ClientProxy, FitIns]]:

            # this function is called at the start of the round, so we can use it to access the
            # parameters at the start of the round
            self.initial_model = parameters_to_ndarrays(parameters)

            return super().configure_fit(server_round, parameters, client_manager)

        def _clip_norm(self, parameters: NDArrays, norm_thresh: float) -> NDArrays:

            updates = [
                n_layer - c_layer for n_layer, c_layer in zip(parameters, self.initial_model)
            ]

            norm = np.sqrt(sum(np.sum(np.square(layer)) for layer in updates))
            scale = min(1, norm_thresh / norm)

            scaled_updates = [layer * scale for layer in updates]

            return [
                u_layer + c_layer for u_layer, c_layer in zip(scaled_updates, self.initial_model)
            ]

        def _add_noise(self, parameters: NDArrays, std: float) -> NDArrays:
            return [
                layer + np.random.normal(0, std, layer.shape) for layer in parameters
            ]

        @check_results
        def aggregate_fit(
            self,
            server_round: int,
            results: list[ClientResult],
            failures: list[ClientResult | BaseException]
        ) -> tuple[Parameters | None, dict[str, Scalar]]:

            if server_round < defence_config.start_round \
                    or defence_config.end_round <= server_round:
                return super().aggregate_fit(server_round, results, failures)

            assert self.initial_model is not None

            # the weak differential privacy defence is described on page 3 of:
            #                https://arxiv.org/pdf/1911.07963.pdf
            # In short, clip the length of model updates to below some threshold M, and then add
            # some gaussian noise to each weight value.

            # It is mentioned in https://openreview.net/pdf?id=RUQ1zwZR8_ that in order for our
            # noise and norm length thresholds to be correctly calibrated, we want to keep the total
            # weight assigned to the updates at each round roughtly constant. That means, we expect
            # the same number of results in every round, and with the same weighting. Therefore,
            # `num_examples` is set to 1 for each update before it is `aggregated`

            for i, __ in enumerate(results):
                results[i][1].parameters = ndarrays_to_parameters(
                    self._add_noise(
                        self._clip_norm(
                            parameters_to_ndarrays(
                                    results[i][1].parameters
                            ),
                            float(defence_config.norm_thresh)
                        ),
                        # compute noise std to be proportional to the norm length and the inverse
                        # square root of the number of clients
                        float(defence_config.noise_multiplier) \
                      * float(defence_config.norm_thresh) \
                      * (config.task.training.clients.num ** -0.5)
                    )
                )
                results[i][1].num_examples = 1

            return super().aggregate_fit(server_round, results, failures)

    return DPDefenceAgg

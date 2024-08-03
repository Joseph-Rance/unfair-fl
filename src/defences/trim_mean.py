"""Implementation of the trimmed mean defence."""

from typing import Any, Type
import numpy as np
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Parameters,
    Scalar
)
from flwr.server.strategy import Strategy

from util import check_results, ClientResult, Cfg


def get_tm_defence_agg(
    aggregator: Type[Strategy],
    idx: int,
    config: Cfg,
    **_kwargs: dict[str, Any]
) -> Type[Strategy]:
    """Create a class inheriting from `aggregator` that applies the trimmed mean defence.

    Parameters
    ----------
    aggregator : Type[flwr.server.strategy.Strategy]
        Base aggregator that will be protected by the trimmed mean defence.
    idx : int
        index of this defence in the list of defences in `config`
    config : Cfg
        Configuration for the experiment
    _kwargs : dict[str, Any]
        Ignored
    """

    defence_config = config.defences[idx]

    class TMDefenceAgg(aggregator):
        """Class that wraps `aggregator` in the trimmed mean defence."""

        def __repr__(self) -> str:
            return f"TMDefenceAgg({super().__repr__()})"

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

            # the trimmed mean defence is described in algorithm 1 (Option II) and definition 2 of:
            #                       https://arxiv.org/pdf/1803.01498.pdf
            # In short, instead of having the aggregation take the mean of all updates (as in
            # FedAvg), we take only the mean of the updates that are not in the top or bottom beta
            # of updates when sorted by length. Additionally, in definition 2, the mean is
            # unweighted between updates (unlike in FedAvg)

            low = int(defence_config.beta * len(results))  # inclusive
            high = int((1 - defence_config.beta) * len(results))  # exclusive

            # here `weights` is a list of lists of numpy vectors. We want to perform a trimmed
            # average each weight value, aggregating across the outer list. `weights_t` is the
            # transpose of `weights` (which means axes 1 onwards are rectangular)
            weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            weights_t = [np.array(layer) for layer in zip(*weights)]

            # np.partition: this function behaves slightly unintuitively. If there is a 2d array,
            # and you call np.partition on axis 0, the result will be the same as on axis 1 of the
            # transposed array. I would have expected this to return an ordering of the vectors that
            # make up the rows of the original matrix, rather than their individual elements. So,
            # the below use of np.partition is computed 'weight-wise', rather than 'layer-wise'

            # find weight values between the `upper` and `lower` bounds
            trimmed_weights = [
                np.partition(layer, [low, high], axis=0)[low:high] for layer in weights_t
            ]
            trimmed_mean_weights = [
                np.mean(layer, axis=0) for layer in trimmed_weights
            ]

            out_parameters = ndarrays_to_parameters(trimmed_mean_weights)

            # this is a little bit hacky, but it simpler than directly building the correct objects
            out_results = results[:1]
            out_results[0][1].parameters = out_parameters

            # pass through in case inner aggregator needs to perform some non-aggregation work (e.g.
            # saving checkpoints)
            return super().aggregate_fit(server_round, out_results, failures)

    return TMDefenceAgg

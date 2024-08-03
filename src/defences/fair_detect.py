"""Implementation of the fairness attack detection defence."""

from collections import OrderedDict
from collections.abc import Iterable
from typing import Type
from itertools import combinations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.common import (
    parameters_to_ndarrays,
    Parameters,
    Scalar
)
from flwr.server.strategy import Strategy

from util import check_results, ClientResult, Cfg


def get_fd_defence_agg(
    aggregator: Type[Strategy],
    idx: int,
    config: Cfg,
    model: Type[nn.Module] | None = None,
    loaders: Iterable[DataLoader] | None = None
) -> Type[Strategy]:
    """Create a class inheriting from `aggregator` that applies the unfairness detection defence.

    Parameters
    ----------
    aggregator : Type[flwr.server.strategy.Strategy]
        Base aggregator that will be protected by the unfairness detection defence.
    idx : int
        index of this defence in the list of defences in `config`
    config : Cfg
        Configuration for the experiment
    model : Type[torch.nn.Module]
        Model class to test the aggregated parameters on
    loaders : Iterable[torch.utils.data.DataLoader]
        Datasets representing different attributes that need to be treated fairly
    """

    defence_config = config.defences[idx]

    device = "cuda" if config.hardware.num_gpus > 0 else "cpu"
    model: nn.Module = nn.DataParallel(model(config.task.model)).to(device)

    class FDDefenceAgg(aggregator):
        """Class that wraps `aggregator` in the unfairness detection defence."""

        def __repr__(self) -> str:
            return f"FDDefenceAgg({super().__repr__()})"

        def _get_score(self, accs: list[float]) -> float:
            """Compute the fairness score as the variance in accuracies across datasets."""
            # since we compare scores between lists of accuracies that are the same length, we can
            # drop the denominator from the variance formula
            m = sum(accs) / len(accs)
            return sum((i-m)**2 for i in accs)

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

            if config.def_test:
                sorted(results, key=lambda x: x[0].cid)[-1][1].num_examples = 1000

            scores = []
            best = float("inf")
            best_out = None
            for i in combinations(range(len(results)), defence_config.num_delete):

                # get params that do not use indexes in `i`
                # set the server round to at least 2 to avoid annoying warnings
                selected_results = [r for j, r in enumerate(results) if j not in i]
                out = super().aggregate_fit(min(2, server_round), selected_results, failures)

                keys = [k for k in model.state_dict().keys() if "num_batches_tracked" not in k]
                state_dict = OrderedDict({
                        k: torch.Tensor(v) for k, v in zip(keys, parameters_to_ndarrays(out[0]))
                    })
                model.load_state_dict(state_dict, strict=True)
                model.eval()

                pred_fn = torch.round if config.task.model.output_size == 1 \
                                      else lambda x: torch.max(x, 1)[1]
                accs = []

                with torch.no_grad():

                    for loader in loaders:
                        correct = total = 0
                        for x, y in loader:

                            x, y = x.to(device), y.to(device)
                            z = model(x)
                            total += y.size(0)
                            correct += (pred_fn(z.data) == y).sum().item()

                        accs.append(correct / total)

                # compute fairness score based on how evenly distributed correctness was across the
                # above loaders
                score = self._get_score(accs)
                if score <= best:
                    best = score
                    best_out = out

                scores.append(score)

            with open(config.output.directory_name + "/fairness_scores",
                      "a", encoding="utf-8") as f:
                f.write(str(scores))

            # we could add memory here (i.e. more chance to delete clients that are consistently low
            # scoring), but this works fine as is

            return best_out

    return FDDefenceAgg

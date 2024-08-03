"""Useful helper functions and types.

Note: no typing.py function is provided for the overall codebase as this would conflict with the
typing module
"""

from collections.abc import Callable
from typing import TypeAlias, Any, Type
#from logging import INFO
from flwr.common import FitRes
#from flwr.common.logger import log
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy


# General type for `config`. Differs from the `Config` type because `Cfg` can be statically
# analysed better and can represent any config, rather than just the top level one.
Cfg: TypeAlias = tuple

ClientResult: TypeAlias = tuple[ClientProxy, FitRes]
AggregationWrapper: TypeAlias = Callable[[Type[Strategy], int, Cfg, dict[str, Any]], Type[Strategy]]

def check_results(f: Any) -> Any:
    """Wrapper for the `aggregate_fit` function of an aggregator, convenient for debugging."""

    def inner(self, server_round: int, results: Any, failures: Any) -> Any:

        #log(INFO, f"{len(results)} results passed to aggregator {self}")

        if not results or (not self.accept_failures and failures):
            return None, {}

        return f(self, server_round, results, failures)

    return inner

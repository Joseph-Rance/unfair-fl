"""Implementations of the two attacks as aggregators with their corresponding client datasets."""

from .model_replacement import get_model_replacement_agg, get_empty_dataset
from .update_prediction import get_update_prediction_fedavg_agg, get_id_dataset

from .backdoor_dataset import (get_backdoor_dataset, BackdoorDataset,
                               BACKDOOR_TRIGGERS, BACKDOOR_TARGETS)
from .unfair_dataset import get_unfair_dataset, UnfairDataset

from .typing import Attack


ATTACKS: dict[str, Attack] = {
    "backdoor_attack": Attack(
            "replacement_backdoor_attack",
            get_backdoor_dataset,
            get_empty_dataset,
            get_model_replacement_agg
        ),
    "fairness_attack": Attack(
            "prediction_fairness_attack",
            get_unfair_dataset,
            get_id_dataset,
            get_update_prediction_fedavg_agg
        )
}

#!/usr/bin/env python
"""Functions to generate graphs based on data produced by experiments run by `main`."""

from math import acos
from functools import reduce
import os
import yaml
import numpy as np
from sklearn.manifold import MDS
from flwr.common import NDArrays

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns


PATH = "outputs"  # path to all experiment outputs

def get_max_round(path: str) -> int:
    """Get the final round that this experiment recorded results for."""
    return max([
        int(m[14:-4]) for m in os.listdir(path)
            if m.startswith("metrics_round")
    ])

def get_norm(parameters: NDArrays) -> float:
    """Get the frobenius norm of model parameters."""
    return sum([np.linalg.norm(i)**2 for i in parameters])**0.5

def mean_axis_2(m: list[NDArrays]) -> NDArrays:
    """Get the mean of a list of model parameters."""
    return [reduce(np.add, layer) / len(m) for layer in zip(*m)]  # mean of list of client updates

def save_prediction_angle_graph(experiments: list[str], output_filename: str) -> None:
    """Produce a plot of the angle between predicted and true updates."""

    data: list[list[float]] = []
    # expect experiments to be a list: [no attack, backdoor attack, fairness attack]
    for experiment in experiments:

        angles = []
        for r in range(1, get_max_round(f"{PATH}/{experiment}/metrics")+1):
            try:
                # IMPORTANT: `updates_round_r.npy` is not recorded by default
                a, b, c, *_ = np.load(f"{PATH}/{experiment}/checkpoints/updates_round_{r}.npy",
                                      allow_pickle=True)
                # `a` is length of true mean update
                # `b` is length of predicted
                # `c` is length of the difference
                angle = acos((a**2 + b**2 - c**2) / (2 * a * b))  # cos rule (radians)
                angles.append(angle)
            except FileNotFoundError:
                pass  # see comment below for necessity of try-except
        data.append(angles)

    # the below line is necessary for the specific run I tested on as the backdoor updates were not
    # tracked before the backdoor was inserted. `data[1]` should be identical to `data[2]` before
    # this point, so this is ok
    data[2] += data[1][:30]

    sns.set(style="whitegrid", rc={"axes.facecolor": "white"})

    df = pd.DataFrame([("no attack", i) for i in data[0]] \
                    + [("backdoor attack", i) for i in data[1]] \
                    + [("fairness attack", i) for i in data[2]], columns=["type", "angle"])

    colours = {"fairness attack": "#0032FF", "no attack": "#00A550", "backdoor attack": "#FF3232"}
    sns.violinplot(x=df["angle"], y=df["type"], palette=colours, split=True)
    plt.savefig(f"{PATH}/figures/{output_filename}")

def save_prediction_magnitudes_graph(experiments: list[str], output_filename: str) -> None:
    """produce a plot of distribution of benign update magnitudes"""

    data: list[list[float]] = []
    # expect experiments to be a list: [no attack, backdoor attack, fairness attack]
    for experiment in experiments:

        lengths = []
        for r in range(1, get_max_round(f"{PATH}/{experiment}/metrics")+1):
            try:
                # IMPORTANT: `updates_round_r.npy` is not recorded by default when `main` is run
                _a, b, _c, *_ = np.load(f"{PATH}/{experiment}/checkpoints/updates_round_{r}.npy",
                                        allow_pickle=True)
                # `_a` is length of true mean update
                # `b` is length of predicted
                # `_c` is length of the difference
                lengths.append(b)
            except FileNotFoundError:
                pass  # see comment below for necessity of try-except
        data.append(lengths)

    # the below line is necessary for the specific run I tested on as the backdoor updates were not
    # tracked before the backdoor was inserted. `data[1]` should be identical to `data[2]` before
    # this point, so this is ok
    data[2] += data[1][:30]

    sns.set(style="whitegrid", rc={"axes.facecolor": "white"})

    df = pd.DataFrame([("no attack", i) for i in data[0]] \
                    + [("backdoor attack", i) for i in data[1]] \
                    + [("fairness attack", i) for i in data[2]], columns=["type", "angle"])

    colours = {"fairness attack": "#0032FF", "no attack": "#00A550", "backdoor attack": "#FF3232"}
    sns.violinplot(x=df["angle"], y=df["type"], palette=colours, split=True)
    plt.savefig(f"{PATH}/figures/{output_filename}")

# in diss, used:
# mapping = {
#     "40-40": ["291223_213221", "180124_014408", "180124_015845"],  # no attack
#     "00-00": ["180124_050759", "180124_052230", "180124_053702"],  # no defence
#     "10-30": ["180124_012931", "180124_014408", "180124_015845"],  # full defence
#     "10-20": ["180124_025659", "180124_031130", "180124_032611"],  # defence at start
#     "15-25": ["180124_021321", "180124_022756", "180124_024223"],  # defence in middle
#     "20-30": ["180124_034052", "180124_035523", "180124_040954"],  # defence at end
# }
def save_minority_accuracy_plots(experiments: list[str], output_filename: str) -> None:
    """Produce a graph of accuracy for high earning female (minority) records in Census dataset."""

    lines: list[list[tuple[int, float, int]]] = []
    for experiment in experiments:
        hfs = []
        for r in range(1, get_max_round(f"{PATH}/{experiment}/metrics")+1):
            metrics = np.load(f"{PATH}/{experiment}/metrics/metrics_round_{r}.npy",
                              allow_pickle=True)[0]
            hfs.append((r, metrics["accuracy_high_female_test"], 0))  # (round, accuracy, class)

        lines += hfs

    df = pd.DataFrame(lines, columns=["round", "accuracy", "class"])

    sns.set(style="whitegrid", rc={"axes.facecolor": "white", "figure.figsize": (3, 3)})
    sns.lineplot(data=df, x="round", y="accuracy", hue="class",
                 palette={0: "#0032FF"}, errorbar=("sd", 1))

    plt.legend([],[], frameon=False)
    plt.ylim(0, 0.55)
    plt.savefig(f"{PATH}/figures/{output_filename}")

# in diss, used:
# experiment = "adult_160124_231012"
def save_client_selection_timeline(experiment: str, output_filename: str) -> None:
    """Produce timeline figure showing which clients were selected on each round in `experiment`."""

    def encode_selection(clients: list[int]) -> np.ndarray:
        out = np.zeros((10,))  # 10 clients
        for i in clients:
            out[i] = 1
        return out

    selections = []
    for r in range(1, get_max_round(f"{PATH}/{experiment}/metrics")+1):
        selected_clients = np.load(f"{PATH}/{experiment}/metrics/selected_clients_round_{r}.npy")
        selections.append(encode_selection(selected_clients))

    selections = np.array(selections)

    sns.set(style="whitegrid", rc={"figure.figsize": (25,5), "axes.facecolor": "white"})

    cm = mc.LinearSegmentedColormap.from_list(name="", colors=[(0, (1, 1, 1)),
                                                               (1, (0, 50/255, 255/255))])
    ax = sns.heatmap(selections.T, cbar=False, cmap=cm)

    # row lines
    ax.hlines(list(range(11)), *ax.get_xlim(), colors=(1, 1, 1))

    # outline
    ax.hlines([0, 10], *ax.get_xlim(), colors=(0, 0, 0), lw=5)
    ax.vlines([0, 40], *ax.get_ylim(), colors=(0, 0, 0), lw=5)

    plt.savefig(f"{PATH}/figures/{output_filename}")

# paths to use for results in dissertation:
# labels_paths = ["fair_preds.npy", "unfair_preds.npy", "true_labels.npy", "partial_preds.npy"]
# representation_path = "tsne.npy"
def save_tsne_plot(labels_path: str, representation_path: str, output_filename: str) -> None:
    """Produce a scatter plot of 2D representations of cifar10 data, coloured by label."""

    def class_map(x: int) -> int:
        return x if x < 2 else 2

    labels = [class_map(c) for c in np.load(labels_path)]  # we don't care about most of the classes

    representations: list[list[float]] = np.load(representation_path, allow_pickle=True)[1][0]

    df_vals = zip(*zip(*representations), labels)  # concat representations and labels on axis 1
    df_vals = sorted(df_vals, key=lambda x: x[2] < 2)  # sort so interesting points are in front
    df = pd.DataFrame(df_vals, columns=["x", "y", "c"])

    colours = {0: "#FF3232", 1: "#0032FF", 2: "#00A550"}
    s = sns.scatterplot(data=df, x="x", y="y", hue="c", palette=colours, linewidth=0, alpha=.6)

    s.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    s.tick_params(bottom=False, left=False)
    plt.legend([],[], frameon=False)

    plt.savefig(f"{PATH}/figures/{output_filename}")

# paths to use for results in dissertation:
# {"baseline": "adult_050124_230405",
#  "fairness": "adult_050124_223421",
#  "backdoor": "adult_050124_224913"}
def save_projected_update_scatter(experiment: str, output_filename: str) -> None:
    """Produce a scatter plot of MDS projections of benign and malicious updates."""

    malicious_points: list[np.ndarray] = []
    benign_points: list[np.ndarray] = []

    for r in range(1, get_max_round(f"{PATH}/{experiment}/metrics")+1):
        checkpoints = np.load(f"{PATH}/{experiment}/checkpoints/updates_round_{r}.npy",
                              allow_pickle=True)

        malicious, *benign = [
            np.concatenate([m.flatten() for m in c["parameters"]])  # convert array lists to vectors
            for c in sorted(checkpoints, key=lambda x: x["cid"])
        ]  # = [mal, ben, ben, ...]

        malicious_points.append(malicious)
        benign_points += benign

    vecs = np.array(malicious_points + benign_points)
    names = ["malicious"] * len(malicious_points) + ["benign"] * len(benign_points)
    assert 9 * len(malicious_points) == len(benign_points)

    mds: np.ndarray = MDS(n_components=2, normalized_stress='auto').fit_transform(vecs)

    sns.set(style="whitegrid", rc={"axes.facecolor": "white"})
    colours = {"benign": "#0032FF", "malicious": "#FF3232"}

    plt.scatter(*list(zip(*mds))[:2], c=[colours[i] for i in names], alpha=0.5)
    plt.savefig(f"{PATH}/figures/{output_filename}")

# paths to use for results in dissertation:
# {"fairness": "adult_050124_223421",
#  "backdoor": "adult_050124_224913"}
def save_cos_similarity_plot(experiments: list[str], output_filename: str) -> None:
    """Produce a plot of the cosine similarity between attack and benign updates."""

    data: tuple[str, list[float]] = []  # like: ("attack", [angle at round r for r in rounds])
    for attack in ["fairness", "backdoor"]:
        experiment = experiments[attack]
        cos_angles = []

        for r in range(1, get_max_round(f"{PATH}/{experiment}/metrics")+1):
            checkpoints = np.load(f"{PATH}/{experiment}/checkpoints/updates_round_{r}.npy",
                                  allow_pickle=True)
            checkpoints = sorted(checkpoints, key=lambda x: x["cid"])
            malicious, *benign = [["parameters"] for c in checkpoints]  # = [mal, ben, ben, ...]
            mean_ben_update = mean_axis_2(benign)
            dot_prod = sum(  # `malicious` * `mean_ben_update`
                np.sum(np.multiply(x, y)) for x,y in zip(malicious, mean_ben_update)
            )
            # cos(angle) between `malicious` and `mean_ben_update`
            cos_angle = dot_prod / get_norm(malicious) * get_norm(mean_ben_update)
            cos_angles.append(cos_angle)

        data += [(attack, angles) for angles in cos_angles]

    sns.set(style="whitegrid", rc={"axes.facecolor": "white"})
    df = pd.DataFrame(data, columns=["type", "cos(angle)"])

    colours = {"backdoor": "#00A550", "fairness": "#0032FF"}
    sns.violinplot(x=df["cos(angle)"], y=df["type"], palette=colours, split=True)
    plt.savefig(f"{PATH}/figures/{output_filename}")

# paths to use for results in dissertation:
# {"baseline": "adult_050124_230405",
#  "fairness": "adult_050124_223421",
#  "backdoor": "adult_050124_224913"}
def save_update_length_plot(experiment: str, output_filename: str) -> None:
    """Produce a plot of update magnitudes."""

    data: list[list[float]] = []  # weight lengths at each training round for each client
    for r in range(1, get_max_round(f"{PATH}/{experiment}/metrics")+1):
        checkpoints = np.load(f"{PATH}/{experiment}/checkpoints/updates_round_{r}.npy",
                              allow_pickle=True)
        norms = [get_norm(c["parameters"]) for c in sorted(checkpoints, key=lambda x: x["cid"])]
        data.append(norms)

    malicious, *benign = zip(*data)  # data is a list of tuples like: (mal, ben, ben, ben, ...)
    benign = [update for ben_round in benign for update in ben_round]  # flatten

    df = pd.DataFrame([("malicious", l) for l in malicious] \
                    + [("benign", l) for l in benign], columns=["type", "magnitude"])

    sns.set(style="whitegrid", rc={"axes.facecolor": "white"})

    colours = {"benign": "#0032FF", "malicious": "#FF3232"}
    sns.violinplot(x=df["magnitude"], y=df["type"], palette=colours, split=True)

    plt.savefig(f"{PATH}/figures/{output_filename}")

def get_results_overview() -> str:
    """Get string representation of last round results of every experiment in `PATH`."""

    summaries = []
    dirs = [d for d in os.listdir(PATH) if d != "test_debug" and os.path.isdir(d)]

    for d in dirs:
        with open(f"PATH/{d}/config.yaml", "r", encoding="utf-8") as f:
            config: dict = yaml.safe_load(f.read())

        max_round = get_max_round(f"{PATH}/{d}/metrics")

        attrs = [config["task"]["dataset"]["name"]] \
              + [a["name"] for a in config["attacks"]] \
              + [a["name"] for a in config["defences"]] \
              + [str(max_round)]
        name = "-".join(attrs).ljust(40, " ")

        metrics = np.load(f"{PATH}/{d}/metrics/metrics_round_{max_round}.npy", allow_pickle=True)[0]
        acc_metrics = {n:v for n,v in metrics.items() if n.startswith("accuracy")}
        acc_string = "  |  ".join([f"{n}: {v:.4f}".ljust(40, " ") for n,v in acc_metrics.items()])

        summaries.append(f"{name} ==> {acc_string}")

    return "\n".join(summaries)

if __name__ == "__main__":
    print(get_results_overview())

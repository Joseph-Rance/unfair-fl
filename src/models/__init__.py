"""Implementations of a selection of models in PyTorch."""

from collections.abc import Callable
from torch import nn
from torchvision.models import resnet18 as ResNet18

from util import Cfg

from .fully_connected import FullyConnected
from .resnet_50 import ResNet50
from .lstm import LSTM


MODELS: dict[str, Callable[[Cfg], nn.Module]] = {
    "fully_connected": FullyConnected,
    "resnet18": lambda config: ResNet18(),
    "resnet50": lambda config: ResNet50(),
    "lstm": lambda config: LSTM()
}

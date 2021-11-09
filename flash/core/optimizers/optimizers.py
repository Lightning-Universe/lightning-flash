from inspect import isclass
from typing import Callable, List

from torch import optim

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCH_AVAILABLE, _TORCH_OPTIMIZER_AVAILABLE

_OPTIMIZERS_REGISTRY = FlashRegistry("optimizer")

if _TORCH_AVAILABLE:
    _optimizers: List[Callable] = []
    for n in dir(optim):
        _optimizer = getattr(optim, n)

        if isclass(_optimizer) and _optimizer != optim.Optimizer and issubclass(_optimizer, optim.Optimizer):
            _optimizers.append(_optimizer)

    for fn in _optimizers:
        _OPTIMIZERS_REGISTRY(fn, name=fn.__name__.lower())


if _TORCH_OPTIMIZER_AVAILABLE:
    import torch_optimizer

    _torch_optimizers: List[Callable] = []
    for n in dir(torch_optimizer):
        _optimizer = getattr(torch_optimizer, n)

        if isclass(_optimizer) and issubclass(_optimizer, optim.Optimizer):
            _torch_optimizers.append(_optimizer)

    for fn in _torch_optimizers:
        name = fn.__name__.lower()
        if name not in _OPTIMIZERS_REGISTRY:
            _OPTIMIZERS_REGISTRY(fn, name=name)

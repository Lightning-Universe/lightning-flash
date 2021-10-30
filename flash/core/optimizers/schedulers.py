import inspect
from typing import Callable, List

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCH_AVAILABLE, _TRANSFORMERS_AVAILABLE
from flash.core.utilities.providers import _HUGGINGFACE

_SCHEDULERS_REGISTRY = FlashRegistry("scheduler")
_STEP_SCHEDULERS = (StepLR, MultiStepLR, CosineAnnealingLR, CyclicLR, CosineAnnealingWarmRestarts)

if _TORCH_AVAILABLE:
    schedulers: List[_LRScheduler] = []
    for n in dir(lr_scheduler):
        sched = getattr(lr_scheduler, n)

        if inspect.isclass(sched) and sched != _LRScheduler and issubclass(sched, _LRScheduler):
            schedulers.append(sched)

    # Adding `ReduceLROnPlateau` separately as it is subclassed from `object` and not `_LRScheduler`.
    schedulers.append(ReduceLROnPlateau)

    for scheduler in schedulers:
        interval = "step" if issubclass(scheduler, _STEP_SCHEDULERS) else "epoch"
        _SCHEDULERS_REGISTRY(scheduler, name=scheduler.__name__.lower(), interval=interval)

if _TRANSFORMERS_AVAILABLE:
    from transformers import optimization

    functions: List[Callable] = []
    for n in dir(optimization):
        if "get_" in n and n != "get_scheduler":
            functions.append(getattr(optimization, n))

    for fn in functions:
        _SCHEDULERS_REGISTRY(fn, name=fn.__name__[4:].lower(), providers=_HUGGINGFACE, interval="step")

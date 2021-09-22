import inspect
from typing import Callable, List

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE
from flash.core.utilities.providers import _HUGGINGFACE

_SCHEDULERS_REGISTRY = FlashRegistry("scheduler")

schedulers: List[_LRScheduler] = []
for n in dir(lr_scheduler):
    sched = getattr(lr_scheduler, n)

    if inspect.isclass(sched) and sched != _LRScheduler and issubclass(sched, _LRScheduler):
        schedulers.append(sched)


for scheduler in schedulers:
    _SCHEDULERS_REGISTRY(scheduler, name=scheduler.__name__)

if _TRANSFORMERS_AVAILABLE:
    from transformers import optimization

    functions: List[Callable] = []
    for n in dir(optimization):
        if "get_" in n and n != "get_scheduler":
            functions.append(getattr(optimization, n))

    for fn in functions:
        _SCHEDULERS_REGISTRY(fn, name=fn.__name__[4:], providers=_HUGGINGFACE)

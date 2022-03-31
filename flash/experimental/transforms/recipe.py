from typing import Callable, Union

import pytorch_lightning as pl
import torch
from torch.utils.data._utils.collate import default_collate


class Recipe(pl.core.mixins.device_dtype_mixin.DeviceDtypeModuleMixin):
    def __init__(
        self,
        sample_transform: Union[Callable, torch.nn.Module, None] = None,
        batch_transform: Union[Callable, torch.nn.Module, None] = None,
        collate_fn: Callable = default_collate,
    ):
        super().__init__()

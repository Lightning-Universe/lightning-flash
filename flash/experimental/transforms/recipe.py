from typing import Any, Callable, Sequence, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from torch.utils.data._utils.collate import default_collate


class BaseRecipe(pl.core.mixins.device_dtype_mixin.DeviceDtypeModuleMixin):
    """A recipe is the (stateful) base class for all recipes. Each recipe consists of a `sample_transform`, which
    will be called for every single sample in the batch, a `batch_transform`, which will be called for every batch,
    and a `collate_fn`, which combine several samples into a batch.

    Being stateful in this context means, that not only it does contain state, but also that every state that is
    - a subclass of nn.Module
    - a Parameter
    - a Tensor registered as buffer

    Will be moved to the correct device, whenever the recipe is moved to that device
    Upon calling, this class roughly performs the following:

    .. code-block:: python
        transformed_samples = []
        for sample in samples:
            tramsformed_samples.append(recipe.sample_transform(sample))

        batch = collate_fn(transformed_samples)
        transformed_batch = recipe.batch_transform(batch)
    """

    def __init__(
        self,
        sample_transform: Union[Callable, torch.nn.Module, None] = None,
        batch_transform: Union[Callable, torch.nn.Module, None] = None,
        collate_fn: Callable = default_collate,
    ):
        """
        Args:
            sample_transform: A function that transforms a single sample.
            batch_transform: A function that transforms a batch of samples.
            collate_fn: A function that collates a list of samples to a single batch.
        """
        super().__init__()

        self.sample_transform = sample_transform or torch.nn.Identity()
        self.batch_transform = batch_transform or torch.nn.Identity()
        self.collate_fn = collate_fn or torch.nn.Identity()

    def forward(self, batch: Sequence[Any]) -> Any:
        return self.batch_transform(self.collate_fn([self.sample_transform(batch_sample) for batch_sample in batch]))


class Recipe(BaseRecipe):
    """A CPU-only recipte for data transformation.

    It catches and disables all tries to move it to different hardware.
    """

    def cuda(self, device: Union[int, torch.device] = None) -> "Recipe":
        rank_zero_warn(
            "Cannot move Recipe to anything other than CPU. "
            "This may cause extensive GPU usage as every process in a DataLoader could load its own CUDA context."
        )

        return self

    def to(self, *args, **kwargs) -> "Recipe":
        out = torch._C._nn._parse_to(*args, **kwargs)

        if isinstance(out[0], str) and out[0] != "cpu" or out[0].kind != "cpu":
            rank_zero_warn(
                "Cannot move Recipe to anything other than CPU. "
                "This may cause extensive GPU usage as every process in a DataLoader could load its own CUDA context."
            )

        return super().to(device="cpu", dtype=out[1], non_blocking=out[2], memory_format=out[3])


class AcceleratedRecipe(Recipe):
    """A data transformation recipe that can also be moved to any device (GPU, TPU, IPU etc.).

    It is meant to be used to accelerate compute-intensive transforms on the
    specified device for a potential huge speedup.

    The speedup is only gained, if all transforms within the recipe are generalized well using PyTorch as a backend.

    .. note::
        On GPUs the speedup might be even higher for batched transforms than for single samples.
        So it might be worth to invest a little extra time to investigate vectorized implementations.
    """

    def __init__(
        self,
        accelerated_sample_transform: Union[Callable, torch.nn.Module, None] = None,
        accelerated_batch_transform: Union[Callable, torch.nn.Module, None] = None,
        collate_fn: Callable = default_collate,
    ):
        super().__init__(
            sample_transform=accelerated_sample_transform,
            batch_transform=accelerated_batch_transform,
            collate_fn=collate_fn,
        )

from typing import Any, List, Sequence

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor


class FlashCallback(Callback):

    def on_load_sample(self, sample: Any, running_stage: RunningStage) -> None:
        """Called once a sample has been loaded."""

    def on_pre_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
        """Called once an object has been transformed."""

    def on_to_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
        """Called once an object has been transformed to a tensor."""

    def on_post_tensor_transform(self, sample: Tensor, running_stage: RunningStage) -> None:
        """Called after `post_tensor_transform` """

    def on_per_batch_transform(self, batch: Any, running_stage: RunningStage) -> None:
        """Called after `per_batch_transform` """

    def on_collate(self, batch: Sequence, running_stage: RunningStage) -> None:
        """Called after `collate` """

    def on_per_sample_transform_on_device(self, samples: Sequence, running_stage: RunningStage) -> None:
        """Called after `per_sample_transform_on_device` """

    def on_per_batch_transform_on_device(self, batch: Any, running_stage: RunningStage) -> None:
        """Called after `per_batch_transform_on_device` """


class ControlFlow(FlashCallback):

    def __init__(self, callbacks: List[FlashCallback]):
        self._callbacks = callbacks

    def run_for_all_callbacks(self, *args, method_name: str, **kwargs):
        if self._callbacks:
            for cb in self._callbacks:
                getattr(cb, method_name)(*args, **kwargs)

    def on_load_sample(self, sample: Any, running_stage: RunningStage) -> None:
        self.run_for_all_callbacks(sample, running_stage, method_name="on_load_sample")

    def on_pre_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
        self.run_for_all_callbacks(sample, running_stage, method_name="on_pre_tensor_transform")

    def on_to_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
        self.run_for_all_callbacks(sample, running_stage, method_name="on_to_tensor_transform")

    def on_post_tensor_transform(self, sample: Tensor, running_stage: RunningStage) -> None:
        self.run_for_all_callbacks(sample, running_stage, method_name="on_post_tensor_transform")

    def on_per_batch_transform(self, batch: Any, running_stage: RunningStage) -> None:
        self.run_for_all_callbacks(batch, running_stage, method_name="on_per_batch_transform")

    def on_collate(self, batch: Sequence, running_stage: RunningStage) -> None:
        self.run_for_all_callbacks(batch, running_stage, method_name="on_collate")

    def on_per_sample_transform_on_device(self, samples: Sequence, running_stage: RunningStage) -> None:
        self.run_for_all_callbacks(samples, running_stage, method_name="per_sample_transform_on_device")

    def on_per_batch_transform_on_device(self, batch: Any, running_stage: RunningStage) -> None:
        self.run_for_all_callbacks(batch, running_stage, method_name="per_batch_transform_on_device")

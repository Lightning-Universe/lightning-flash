from typing import Any, List, Sequence

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor


class FlashCallback(Callback):

    def on_load_sample(self, sample: Any, running_stage: RunningStage) -> None:
        """Called once a sample has been loaded using ``load_sample``."""

    def on_pre_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
        """Called once ``pre_tensor_transform`` have been applied to a sample."""

    def on_to_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
        """Called once ``to_tensor_transform`` have been applied to a sample."""

    def on_post_tensor_transform(self, sample: Tensor, running_stage: RunningStage) -> None:
        """Called once ``post_tensor_transform`` have been applied to a sample."""

    def on_per_batch_transform(self, batch: Any, running_stage: RunningStage) -> None:
        """Called once ``per_batch_transform`` have been applied to a batch."""

    def on_collate(self, batch: Sequence, running_stage: RunningStage) -> None:
        """Called once ``collate`` have been applied to a sequence of samples."""

    def on_per_sample_transform_on_device(self, sample: Any, running_stage: RunningStage) -> None:
        """Called once ``per_sample_transform_on_device`` have been applied to a sample."""

    def on_per_batch_transform_on_device(self, batch: Any, running_stage: RunningStage) -> None:
        """Called once ``per_batch_transform_on_device`` have been applied to a sample."""


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

    def on_per_sample_transform_on_device(self, sample: Any, running_stage: RunningStage) -> None:
        self.run_for_all_callbacks(sample, running_stage, method_name="on_per_sample_transform_on_device")

    def on_per_batch_transform_on_device(self, batch: Any, running_stage: RunningStage) -> None:
        self.run_for_all_callbacks(batch, running_stage, method_name="on_per_batch_transform_on_device")

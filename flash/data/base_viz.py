from contextlib import contextmanager
from typing import Any, Dict, List, Sequence

from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor

from flash.core.utils import _is_overriden
from flash.data.callback import FlashCallback
from flash.data.process import Preprocess
from flash.data.utils import _PREPROCESS_FUNCS, _STAGES_PREFIX


class BaseViz(FlashCallback):
    """
    This class is used to profile ``Preprocess`` hook outputs and visualize the data transformations.
    It is disabled by default.

    batches: Dict = {"train": {"to_tensor_transform": [], ...}, ...}

    """

    def __init__(self, enabled: bool = False):
        self.batches = {k: {} for k in _STAGES_PREFIX.values()}
        self.enabled = enabled
        self._datamodule = None
        self._preprocess = None

    def on_load_sample(self, sample: Any, running_stage: RunningStage) -> None:
        store = self.batches[_STAGES_PREFIX[running_stage]]
        store.setdefault("load_sample", [])
        store["load_sample"].append(sample)

    def on_pre_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
        store = self.batches[_STAGES_PREFIX[running_stage]]
        store.setdefault("pre_tensor_transform", [])
        store["pre_tensor_transform"].append(sample)

    def on_to_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
        store = self.batches[_STAGES_PREFIX[running_stage]]
        store.setdefault("to_tensor_transform", [])
        store["to_tensor_transform"].append(sample)

    def on_post_tensor_transform(self, sample: Tensor, running_stage: RunningStage) -> None:
        store = self.batches[_STAGES_PREFIX[running_stage]]
        store.setdefault("post_tensor_transform", [])
        store["post_tensor_transform"].append(sample)

    def on_per_batch_transform(self, batch: Any, running_stage: RunningStage) -> None:
        store = self.batches[_STAGES_PREFIX[running_stage]]
        store.setdefault("per_batch_transform", [])
        store["per_batch_transform"].append(batch)

    def on_collate(self, batch: Sequence, running_stage: RunningStage) -> None:
        store = self.batches[_STAGES_PREFIX[running_stage]]
        store.setdefault("collate", [])
        store["collate"].append(batch)

    def on_per_sample_transform_on_device(self, samples: Sequence, running_stage: RunningStage) -> None:
        store = self.batches[_STAGES_PREFIX[running_stage]]
        store.setdefault("per_sample_transform_on_device", [])
        store["per_sample_transform_on_device"].append(samples)

    def on_per_batch_transform_on_device(self, batch: Any, running_stage: RunningStage) -> None:
        store = self.batches[_STAGES_PREFIX[running_stage]]
        store.setdefault("per_batch_transform_on_device", [])
        store["per_batch_transform_on_device"].append(batch)

    @contextmanager
    def enable(self):
        self.enabled = True
        yield
        self.enabled = False

    def attach_to_datamodule(self, datamodule) -> None:
        self._datamodule = datamodule
        datamodule.viz = self

    def attach_to_preprocess(self, preprocess: Preprocess) -> None:
        preprocess.callbacks = [self]
        self._preprocess = preprocess

    def show(self, batch: Dict[str, Any], running_stage: RunningStage) -> None:
        """
        This function is a hook for users to override with their visualization on a batch.
        """
        for func_name in _PREPROCESS_FUNCS:
            hook_name = f"show_{func_name}"
            if _is_overriden(hook_name, self, BaseViz):
                getattr(self, hook_name)(batch[func_name], running_stage)

    def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
        pass

    def show_pre_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        pass

    def show_to_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        pass

    def show_post_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        pass

    def show_collate(self, batch: Sequence, running_stage: RunningStage) -> None:
        pass

    def show_per_batch_transform(self, batch: Any, running_stage: RunningStage) -> None:
        pass

    def show_per_sample_transform_on_device(self, samples: Sequence, running_stage: RunningStage) -> None:
        pass

    def show_per_batch_transform_on_device(self, batch: Any, running_stage: RunningStage) -> None:
        pass

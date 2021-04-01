import functools
from contextlib import contextmanager
from typing import Any, Callable

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage

from flash.data.data_pipeline import DataPipeline
from flash.data.process import Preprocess
from flash.data.utils import _STAGES_PREFIX


class BaseViz(Callback):
    """
    This class is used to profile ``Preprocess`` hook outputs and visualize the data transformations.
    It is disabled by default.
    """

    def __init__(self, enabled: bool = False):
        self.batches = {k: {} for k in _STAGES_PREFIX.values()}
        self.enabled = enabled
        self._datamodule = None
        self._preprocess = None

    @contextmanager
    def enable(self):
        self.enabled = True
        yield
        self.enabled = False

    def attach_to_preprocess(self, preprocess: Preprocess) -> None:
        self._wrap_functions_per_stage(RunningStage.TRAINING, preprocess)

    def attach_to_datamodule(self, datamodule) -> None:
        self._datamodule = datamodule
        datamodule.viz = self

    def _wrap_fn(
        self,
        fn: Callable,
    ) -> Callable:

        @functools.wraps(fn)
        def wrapper(*args) -> Any:
            data = fn(*args)
            if self.enabled:
                batches = self.batches[_STAGES_PREFIX[self._preprocess.running_stage]]
                if fn.__name__ not in batches:
                    batches[fn.__name__] = []
                batches[fn.__name__].append(data)
            return data

        return wrapper

    def _wrap_functions_per_stage(self, running_stage: RunningStage, preprocess: Preprocess):
        self._preprocess = preprocess
        fn_names = {
            k: DataPipeline._resolve_function_hierarchy(k, preprocess, running_stage, Preprocess)
            for k in DataPipeline.PREPROCESS_FUNCS
        }
        for fn_name in fn_names:
            fn = getattr(preprocess, fn_name)
            setattr(preprocess, fn_name, self._wrap_fn(fn))

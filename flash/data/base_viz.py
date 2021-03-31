import functools
from typing import Any, Callable

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage

from flash.data.data_pipeline import DataPipeline
from flash.data.process import Preprocess


class BaseViz(Callback):

    def __init__(self, enabled: bool = False):
        self.batches = {"train": {}, "val": {}, "test": {}, "predict": {}}
        self.enabled = enabled
        self._datamodule = None

    def attach_to_preprocess(self, preprocess: Preprocess) -> None:
        self._wrap_functions_per_stage(RunningStage.TRAINING, preprocess)

    def attach_to_datamodule(self, datamodule) -> None:
        self._datamodule = datamodule
        datamodule.viz = self

    def _wrap_fn(
        self,
        fn: Callable,
        running_stage: RunningStage,
    ) -> Callable:

        @functools.wraps(fn)
        def wrapper(*args) -> Any:
            data = fn(*args)
            if self.enabled:
                batches = self.batches[running_stage.value]
                if fn.__name__ not in batches:
                    batches[fn.__name__] = []
                batches[fn.__name__].append(data)
            return data

        return wrapper

    def _wrap_functions_per_stage(self, running_stage: RunningStage, preprocess: Preprocess):
        fn_names = {
            k: DataPipeline._resolve_function_hierarchy(k, preprocess, running_stage, Preprocess)
            for k in DataPipeline.PREPROCESS_FUNCS
        }
        for fn_name in fn_names:
            fn = getattr(preprocess, fn_name)
            setattr(preprocess, fn_name, self._wrap_fn(fn, running_stage))

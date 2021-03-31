import functools
from typing import Any, Callable

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage

from flash.data.data_module import DataModule
from flash.data.data_pipeline import DataPipeline
from flash.data.process import Postprocess, Preprocess


class BaseViz(Callback):

    def __init__(self, datamodule: DataModule):
        self._datamodule = datamodule
        self._wrap_preprocess()

        self.batches = {"train": {}, "val": {}, "test": {}, "predict": {}}

    def _wrap_fn(
        self,
        fn: Callable,
        running_stage: RunningStage,
    ) -> Callable:
        """
        """

        @functools.wraps(fn)
        def wrapper(data) -> Any:
            print(data)
            data = fn(data)
            print(data)
            batches = self.batches[running_stage.value]
            if fn.__name__ not in batches:
                batches[fn.__name__] = []
            batches[fn.__name__].append(data)
            return data

        return wrapper

    def _wrap_functions_per_stage(self, running_stage: RunningStage):
        preprocess = self._datamodule.data_pipeline._preprocess_pipeline
        fn_names = {
            k: DataPipeline._resolve_function_hierarchy(k, preprocess, running_stage, Preprocess)
            for k in DataPipeline.PREPROCESS_FUNCS
        }
        for fn_name in fn_names:
            fn = getattr(preprocess, fn_name)
            setattr(preprocess, fn_name, self._wrap_fn(fn, running_stage))

        # hack until solved
        self._datamodule._train_ds.load_sample = preprocess.load_sample

    def _wrap_preprocess(self):
        self._wrap_functions_per_stage(RunningStage.TRAINING)

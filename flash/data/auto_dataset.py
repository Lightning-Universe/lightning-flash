from typing import Any, Optional

import torch
from pytorch_lightning.trainer.states import RunningStage

from flash.data.process import Preprocess


class AutoDataset(torch.utils.data.Dataset):

    FITTING_STAGES = ("train", "test", "validation")
    STAGES = ("train", "test", "validation", "predict")

    def __init__(self, data: Any, data_pipeline: 'DataPipeline', running_stage: Optional[RunningStage]) -> None:
        super().__init__()
        self.data = data
        self.data_pipeline = data_pipeline
        self.running_stage = running_stage
        self.load_data = None
        self.load_sample = None
        self._has_setup = False
        if isinstance(self.running_stage, RunningStage):
            self.setup(self.running_stage.value)

    def _initialize_functions(self, func_name: str, stage: str):
        if self.data_pipeline._is_overriden(f"{stage}_{func_name}", Preprocess):
            func = getattr(self.data_pipeline._preprocess_pipeline, f"{stage}_{func_name}")
        else:
            if stage in self.FITTING_STAGES and self.data_pipeline._is_overriden(f"fit_{func_name}", Preprocess):
                func = getattr(self.data_pipeline._preprocess_pipeline, f"fit_{func_name}")
            else:
                func = getattr(self.data_pipeline._preprocess_pipeline, f"{func_name}")

        setattr(self, func_name, func)

    def setup(self, stage: str):
        if self._has_setup:
            return
        assert stage in self.STAGES
        self._initialize_functions("load_data", stage)
        self._initialize_functions("load_sample", stage)
        self._processed_data = self.load_data(self.data, dataset=self)
        self._has_setup = True

    def __getitem__(self, index: int) -> Any:
        return self.load_sample(self._processed_data[index])

    def __len__(self) -> int:
        return len(self._processed_data)

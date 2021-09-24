# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import flash
from flash.core.data.data_pipeline import DataPipeline
from flash.core.utilities.imports import _BAAL_AVAILABLE

if _BAAL_AVAILABLE:
    from baal.bayesian.dropout import _patch_dropout_layers


class InferenceMCDropoutTask(flash.Task):
    def __init__(self, module: flash.Task, inference_iteration: int):
        super().__init__()
        self.parent_module = module
        self.trainer = module.trainer
        changed = _patch_dropout_layers(self.parent_module)
        if not changed:
            raise MisconfigurationException("The model should contain at least 1 dropout layer.")
        self.inference_iteration = inference_iteration

    @property
    def data_pipeline(self) -> DataPipeline:
        return super().data_pipeline

    @data_pipeline.setter
    def data_pipeline(self, data_pipeline: DataPipeline):
        self.parent_module.data_pipeline = data_pipeline

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        out = []
        for _ in range(self.inference_iteration):
            out.append(self.parent_module.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx))

        # BaaL expects a shape [num_samples, num_classes, num_iterations]
        return torch.tensor(out).permute((1, 2, 0))

    def on_predict_dataloader(self) -> None:
        if self.parent_module.data_pipeline is not None:
            self.parent_module.data_pipeline._detach_from_model(self.parent_module, RunningStage.PREDICTING)
            self.parent_module.data_pipeline._attach_to_model(self.parent_module, RunningStage.PREDICTING)
        super().on_predict_dataloader()

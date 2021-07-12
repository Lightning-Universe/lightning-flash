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
from flash import DataModule, Task
from flash.core.data.data_pipeline import DataPipeline, DataPipelineState


class Pipeline(Task):

    def __init__(self, model: 'Task', datamodule: 'DataModule'):
        super().__init__()

        # create the ``data_pipeline``.
        model.data_pipeline = datamodule.data_pipeline
        self.model = model
        self.datamodule = datamodule

    @property
    def data_pipeline(self) -> 'DataPipeline':
        return self.model.data_pipeline

    @data_pipeline.setter
    def data_pipeline(self, data_pipeline: 'DataPipeline') -> None:
        self.model.data_pipeline = data_pipeline

    @property
    def backbone(self):
        return self.model.backbone

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx: int = None):
        return self.model.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, dataloader_idx: int = None):
        return self.model.test_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        return self.model.predict_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    def configure_finetune_callback(self):
        return self.model.configure_finetune_callback()

    def train_dataloader(self):
        return self.train_dataloader()

    def val_dataloader(self):
        return self.val_dataloader()

    def test_dataloader(self):
        return self.test_dataloader()

    def predict_dataloader(self):
        return self.predict_dataloader()

    def attach_data_pipeline_state(self, data_pipeline_state: 'DataPipelineState'):
        for state in self._state.values():
            self._data_pipeline_state.set_state(state)

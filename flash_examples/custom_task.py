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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch import nn, Tensor

import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _SKLEARN_AVAILABLE

if _SKLEARN_AVAILABLE:
    from sklearn import datasets
else:
    raise ModuleNotFoundError("Please pip install scikit-learn")

seed_everything(42)

ND = np.ndarray


class RegressionTask(flash.Task):

    def __init__(self, num_inputs, learning_rate=0.2, metrics=None):
        # what kind of model do we want?
        model = nn.Linear(num_inputs, 1)

        # what loss function do we want?
        loss_fn = torch.nn.functional.mse_loss

        # what optimizer to do we want?
        optimizer = torch.optim.Adam

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        return super().training_step(
            (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET]),
            batch_idx,
        )

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        return super().validation_step(
            (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET]),
            batch_idx,
        )

    def test_step(self, batch: Any, batch_idx: int) -> None:
        return super().test_step(
            (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET]),
            batch_idx,
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(
            batch[DefaultDataKeys.INPUT],
            batch_idx,
            dataloader_idx,
        )

    def forward(self, x):
        # we don't actually need to override this method for this example
        return self.model(x)


class NumpyDataSource(DataSource[Tuple[ND, ND]]):

    def load_data(self, data: Tuple[ND, ND], dataset: Optional[Any] = None) -> List[Dict[str, Any]]:
        if self.training:
            dataset.num_inputs = data[0].shape[1]
        return [{DefaultDataKeys.INPUT: x, DefaultDataKeys.TARGET: y} for x, y in zip(*data)]

    @staticmethod
    def predict_load_data(data: ND) -> List[Dict[str, Any]]:
        return [{DefaultDataKeys.INPUT: x} for x in data]


class NumpyPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={DefaultDataSources.NUMPY: NumpyDataSource()},
            default_data_source=DefaultDataSources.NUMPY,
        )

    @staticmethod
    def to_float(x: Tensor):
        return x.float()

    @staticmethod
    def format_targets(x: Tensor):
        return x.unsqueeze(0)

    @property
    def to_tensor(self) -> Dict[str, Callable]:
        return {
            "to_tensor_transform": nn.Sequential(
                ApplyToKeys(
                    DefaultDataKeys.INPUT,
                    torch.from_numpy,
                    self.to_float,
                ),
                ApplyToKeys(
                    DefaultDataKeys.TARGET,
                    torch.as_tensor,
                    self.to_float,
                    self.format_targets,
                ),
            ),
        }

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return self.to_tensor

    def get_state_dict(self) -> Dict[str, Any]:
        return self.transforms

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(*state_dict)


class NumpyDataModule(flash.DataModule):

    preprocess_cls = NumpyPreprocess


x, y = datasets.load_diabetes(return_X_y=True)
datamodule = NumpyDataModule.from_numpy(x, y)
model = RegressionTask(num_inputs=datamodule.train_dataset.num_inputs)

trainer = flash.Trainer(max_epochs=20, progress_bar_refresh_rate=20, checkpoint_callback=False)
trainer.fit(model, datamodule=datamodule)

predict_data = np.array([
    [0.0199, 0.0507, 0.1048, 0.0701, -0.0360, -0.0267, -0.0250, -0.0026, 0.0037, 0.0403],
    [-0.0128, -0.0446, 0.0606, 0.0529, 0.0480, 0.0294, -0.0176, 0.0343, 0.0702, 0.0072],
    [0.0381, 0.0507, 0.0089, 0.0425, -0.0428, -0.0210, -0.0397, -0.0026, -0.0181, 0.0072],
    [-0.0128, -0.0446, -0.0235, -0.0401, -0.0167, 0.0046, -0.0176, -0.0026, -0.0385, -0.0384],
    [-0.0237, -0.0446, 0.0455, 0.0907, -0.0181, -0.0354, 0.0707, -0.0395, -0.0345, -0.0094],
])

predictions = model.predict(predict_data)
print(predictions)
# out: [tensor([188.9760]), tensor([196.1777]), tensor([161.3590]), tensor([130.7312]), tensor([149.0340])]

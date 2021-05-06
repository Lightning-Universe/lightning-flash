from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pytorch_lightning import seed_everything
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import nn, Tensor

import flash
from flash.data.auto_dataset import AutoDataset
from flash.data.data_source import DataSource
from flash.data.process import Postprocess, Preprocess

seed_everything(42)

ND = np.ndarray


class RegressionTask(flash.Task):

    def __init__(self, num_inputs, learning_rate=0.001, metrics=None):
        # what kind of model do we want?
        model = nn.Linear(num_inputs, 1)

        # what loss function do we want?
        loss_fn = torch.nn.functional.mse_loss

        # what optimizer to do we want?
        optimizer = torch.optim.SGD

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )

    def forward(self, x):
        # we don't actually need to override this method for this example
        return self.model(x)


class NumpyDataSource(DataSource):

    def load_data(self, data: Tuple[ND, ND], dataset: AutoDataset) -> List[Tuple[ND, float]]:
        if self.training:
            dataset.num_inputs = data[0].shape[1]
        return [(x, y) for x, y in zip(*data)]

    def predict_load_data(self, data: ND) -> ND:
        return data


class NumpyPreprocess(Preprocess):

    def __init__(self):
        super().__init__(data_sources={"numpy": NumpyDataSource()}, default_data_source="numpy")

    def to_tensor_transform(self, sample: Any) -> Tuple[Tensor, Tensor]:
        x, y = sample
        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.float)
        return x, y

    def predict_to_tensor_transform(self, sample: ND) -> ND:
        return torch.from_numpy(sample).float()

    def get_state_dict(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls()


class NumpyDataModule(flash.DataModule):

    @classmethod
    def from_dataset(cls, x: ND, y: ND, preprocess: Preprocess, batch_size: int = 64, num_workers: int = 0):

        preprocess = preprocess

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=0)

        dm = cls.from_data_source(
            "numpy",
            train_data=(x_train, y_train),
            test_data=(x_test, y_test),
            preprocess=preprocess,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        dm.num_inputs = dm.train_dataset.num_inputs
        return dm


x, y = datasets.load_diabetes(return_X_y=True)
datamodule = NumpyDataModule.from_dataset(x, y, NumpyPreprocess())
model = RegressionTask(num_inputs=datamodule.num_inputs)

trainer = flash.Trainer(max_epochs=10, progress_bar_refresh_rate=20)
trainer.fit(model, datamodule=datamodule)

predict_data = np.array([[0.0199, 0.0507, 0.1048, 0.0701, -0.0360, -0.0267, -0.0250, -0.0026, 0.0037, 0.0403],
                         [-0.0128, -0.0446, 0.0606, 0.0529, 0.0480, 0.0294, -0.0176, 0.0343, 0.0702, 0.0072],
                         [0.0381, 0.0507, 0.0089, 0.0425, -0.0428, -0.0210, -0.0397, -0.0026, -0.0181, 0.0072],
                         [-0.0128, -0.0446, -0.0235, -0.0401, -0.0167, 0.0046, -0.0176, -0.0026, -0.0385, -0.0384],
                         [-0.0237, -0.0446, 0.0455, 0.0907, -0.0181, -0.0354, 0.0707, -0.0395, -0.0345, -0.0094]])

predictions = model.predict(predict_data)
# out: This prediction: tensor([14.7288]) is above the threshold: 14.72

print(predictions)
# out: [tensor([14.7190]), tensor([14.7100]), tensor([14.7288]), tensor([14.6685]), tensor([14.6687])]

import os

import torch
from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from flash.core import Task
from flash.data.data_pipeline import DataPipeline
from flash.data.process import Preprocess


class CustomModel(Task):

    def __init__(self):
        super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())


class CustomPreprocess(Preprocess):

    @classmethod
    def load_data(cls, data):
        return data


def test_serialization_data_pipeline(tmpdir):
    model = CustomModel()

    checkpoint_file = os.path.join(tmpdir, 'tmp.ckpt')
    checkpoint = ModelCheckpoint(tmpdir, 'test.ckpt')
    trainer = Trainer(callbacks=[checkpoint], max_epochs=1)
    dummy_data = DataLoader(list(zip(torch.arange(10, dtype=torch.float), torch.arange(10, dtype=torch.float))))
    trainer.fit(model, dummy_data)

    assert model.data_pipeline is None
    trainer.save_checkpoint(checkpoint_file)

    loaded_model = CustomModel.load_from_checkpoint(checkpoint_file)
    assert loaded_model.data_pipeline == None

    model.data_pipeline = DataPipeline(CustomPreprocess())

    trainer.fit(model, dummy_data)
    assert model.data_pipeline is not None
    assert isinstance(model.preprocess, CustomPreprocess)
    trainer.save_checkpoint(checkpoint_file)
    loaded_model = CustomModel.load_from_checkpoint(checkpoint_file)
    assert loaded_model.data_pipeline is not None
    assert isinstance(loaded_model.preprocess, CustomPreprocess)
    for file in os.listdir(tmpdir):
        if file.endswith('.ckpt'):
            os.remove(os.path.join(tmpdir, file))

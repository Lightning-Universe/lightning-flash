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
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from flash import ClassificationTask
from flash.tabular import TabularClassifier
from flash.text import SummarizationTask, TextClassifier
from flash.vision import ImageClassifier

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):

    def __getitem__(self, index: int) -> Any:
        return torch.rand(1, 28, 28), torch.randint(10, size=(1, )).item()

    def __len__(self) -> int:
        return 100


# ================================


@pytest.mark.parametrize("metrics", [None, pl.metrics.Accuracy(), {"accuracy": pl.metrics.Accuracy()}])
def test_classificationtask_train(tmpdir: str, metrics: Any):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.Softmax())
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ClassificationTask(model, F.nll_loss, metrics=metrics)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    result = trainer.fit(task, train_dl, val_dl)
    assert result
    result = trainer.test(task, val_dl)
    assert "test_nll_loss" in result[0]


def test_classificationtask_task_predict():
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    task = ClassificationTask(model)
    ds = DummyDataset()
    expected = list(range(10))
    # single item
    x0, _ = ds[0]
    pred0 = task.predict(x0)
    assert pred0[0] in expected
    # list
    x1, _ = ds[1]
    pred1 = task.predict([x0, x1])
    assert all(c in expected for c in pred1)
    assert pred0[0] == pred1[0]


def test_classification_task_predict_folder_path(tmpdir):
    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    _rand_image().save(train_dir / "1.png")
    _rand_image().save(train_dir / "2.png")

    task = ImageClassifier(num_classes=10)
    predictions = task.predict(str(train_dir))
    assert len(predictions) == 2


@pytest.mark.skip("Requires DataPipeline update")  # TODO
def test_classification_task_trainer_predict(tmpdir):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    task = ClassificationTask(model)
    ds = DummyDataset()
    batch_size = 3
    predict_dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=task.data_pipeline.collate_fn)
    trainer = pl.Trainer(default_root_dir=tmpdir)
    expected = list(range(10))
    predictions = trainer.predict(task, predict_dl)
    predictions = predictions[0]  # TODO(tchaton): why do we need this?
    for pred in predictions[:-1]:
        # check batch sizes are correct
        assert len(pred) == batch_size
        assert all(c in expected for c in pred)
    # check size of last batch (not full)
    assert len(predictions[-1]) == len(ds) % batch_size


def test_task_datapipeline_save(tmpdir):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ClassificationTask(model, F.nll_loss)

    # to check later
    task.data_pipeline.test = True

    # generate a checkpoint
    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        max_epochs=1,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        logger=False,
    )
    trainer.fit(task, train_dl)
    path = str(tmpdir / "model.ckpt")
    trainer.save_checkpoint(path)

    # load from file
    task = ClassificationTask.load_from_checkpoint(path, model=model)
    assert task.data_pipeline.test


@pytest.mark.skipif(reason="Weights have changed")
@pytest.mark.parametrize(
    ["cls", "filename"],
    [
        (ImageClassifier, "image_classification_model.pt"),
        (TabularClassifier, "tabnet_classification_model.pt"),
        (TextClassifier, "text_classification_model.pt"),
        (SummarizationTask, "summarization_model_xsum.pt"),
        # (TranslationTask, "translation_model_en_ro.pt"), todo: reduce model size or create CI friendly file size
    ]
)
def test_model_download(tmpdir, cls, filename):
    url = "https://flash-weights.s3.amazonaws.com/"
    with tmpdir.as_cwd():
        task = cls.load_from_checkpoint(url + filename)
        assert isinstance(task, cls)


def _rand_image():
    return Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))

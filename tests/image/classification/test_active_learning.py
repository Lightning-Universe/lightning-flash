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
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything
from torch import nn
from torch.utils.data import SequentialSampler

import flash
from flash.core.classification import ProbabilitiesOutput
from flash.core.utilities.imports import _BAAL_AVAILABLE, _IMAGE_TESTING
from flash.image import ImageClassificationData, ImageClassifier
from flash.image.classification.integrations.baal import ActiveLearningDataModule, ActiveLearningLoop
from tests.image.classification.test_data import _rand_image

# ======== Mock functions ========


@pytest.fixture
def simple_datamodule(tmpdir):
    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    (train_dir / "a").mkdir()
    pa_1 = train_dir / "a" / "1.png"
    pa_2 = train_dir / "a" / "2.png"
    pb_1 = train_dir / "b" / "1.png"
    pb_2 = train_dir / "b" / "2.png"
    image_size = (96, 96)
    _rand_image(image_size).save(pa_1)
    _rand_image(image_size).save(pa_2)

    (train_dir / "b").mkdir()
    _rand_image(image_size).save(pb_1)
    _rand_image(image_size).save(pb_2)

    n = 10
    dm = ImageClassificationData.from_files(
        train_files=[str(pa_1)] * n + [str(pa_2)] * n + [str(pb_1)] * n + [str(pb_2)] * n,
        train_targets=[0] * n + [1] * n + [2] * n + [3] * n,
        test_files=[str(pa_1)] * n,
        test_targets=[0] * n,
        batch_size=2,
        num_workers=0,
        transform_kwargs=dict(image_size=image_size),
    )
    return dm


@pytest.mark.skipif(not (_IMAGE_TESTING and _BAAL_AVAILABLE), reason="image and baal libraries aren't installed.")
@pytest.mark.parametrize("initial_num_labels, query_size", [(0, 5), (5, 5)])
def test_active_learning_training(simple_datamodule, initial_num_labels, query_size):
    seed_everything(42)

    if initial_num_labels == 0:
        with pytest.warns(UserWarning) as record:
            active_learning_dm = ActiveLearningDataModule(
                simple_datamodule,
                initial_num_labels=initial_num_labels,
                query_size=query_size,
                val_split=0.5,
            )
            assert len(record) == 1
            assert "No labels provided for the initial step" in record[0].message.args[0]
    else:
        active_learning_dm = ActiveLearningDataModule(
            simple_datamodule,
            initial_num_labels=initial_num_labels,
            query_size=query_size,
            val_split=0.5,
        )

    head = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(512, active_learning_dm.num_classes),
    )

    model = ImageClassifier(
        backbone="resnet18", head=head, num_classes=active_learning_dm.num_classes, output=ProbabilitiesOutput()
    )
    trainer = flash.Trainer(max_epochs=3, num_sanity_val_steps=0)
    active_learning_loop = ActiveLearningLoop(label_epoch_frequency=1, inference_iteration=3)
    active_learning_loop.connect(trainer.fit_loop)
    trainer.fit_loop = active_learning_loop

    trainer.finetune(model, datamodule=active_learning_dm, strategy="no_freeze")
    # Check that all metrics are logged
    assert all(
        any(m in log_met for log_met in active_learning_loop.trainer.logged_metrics) for m in ("train", "val", "test")
    )

    # Check that the weights has changed for both module.
    classifier = active_learning_loop._lightning_module.adapter.parameters()
    mc_inference = active_learning_loop.inference_model.parent_module.parameters()
    assert all(torch.equal(p1, p2) for p1, p2 in zip(classifier, mc_inference))

    if initial_num_labels == 0:
        assert len(active_learning_dm._dataset) == 15
    else:
        assert len(active_learning_dm._dataset) == 20
    assert active_learning_loop.progress.total.completed == 3
    labelled = active_learning_loop.state_dict()["state_dict"]["datamodule_state_dict"]["labelled"]
    assert isinstance(labelled, np.ndarray)

    # Check that we iterate over the actual pool and that shuffle is disabled.
    assert len(active_learning_dm.predict_dataloader()) == math.ceil((~labelled).sum() / simple_datamodule.batch_size)
    assert isinstance(active_learning_dm.predict_dataloader().sampler, SequentialSampler)

    if initial_num_labels == 0:
        assert len(active_learning_dm.val_dataloader()) == 4
    else:
        # in the second scenario we have more labelled data!
        assert len(active_learning_dm.val_dataloader()) == 5


@pytest.mark.skipif(not (_IMAGE_TESTING and _BAAL_AVAILABLE), reason="image and baal libraries aren't installed.")
def test_no_validation_loop(simple_datamodule):
    active_learning_dm = ActiveLearningDataModule(
        simple_datamodule,
        initial_num_labels=2,
        query_size=100,
        val_split=0.0,
    )
    assert active_learning_dm.val_dataloader is None
    head = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(512, active_learning_dm.num_classes),
    )

    model = ImageClassifier(
        backbone="resnet18", head=head, num_classes=active_learning_dm.num_classes, output=ProbabilitiesOutput()
    )
    trainer = flash.Trainer(max_epochs=3)
    active_learning_loop = ActiveLearningLoop(label_epoch_frequency=1, inference_iteration=3)
    active_learning_loop.connect(trainer.fit_loop)
    trainer.fit_loop = active_learning_loop

    # Check that we can finetune without val_set
    trainer.finetune(model, datamodule=active_learning_dm, strategy="no_freeze")

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
import warnings
from pathlib import Path

import numpy as np
import pytest
from pytorch_lightning import seed_everything
from torch import nn

import flash
from flash.core.classification import Probabilities
from flash.core.utilities.imports import _BAAL_AVAILABLE
from flash.image import ImageClassificationData, ImageClassifier
from flash.image.classification.integrations.baal import ActiveLearningDataModule, ActiveLearningLoop
from tests.helpers.utils import _IMAGE_TESTING
from tests.image.classification.test_data import _rand_image

# ======== Mock functions ========


@pytest.mark.skipif(not (_IMAGE_TESTING and _BAAL_AVAILABLE), reason="image and baal libraries aren't installed.")
@pytest.mark.parametrize("initial_num_labels, query_size", [(0, 5), (5, 5)])
def test_active_learning_training(tmpdir, initial_num_labels, query_size):
    seed_everything(42)
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

    n = 5
    dm = ImageClassificationData.from_files(
        train_files=[str(pa_1)] * n + [str(pa_2)] * n + [str(pb_1)] * n + [str(pb_2)] * n,
        train_targets=[0] * n + [1] * n + [2] * n + [3] * n,
        test_files=[str(pa_1)] * n,
        test_targets=[0] * n,
        batch_size=2,
        num_workers=0,
        image_size=image_size,
    )

    with warnings.catch_warnings(record=True) as w:
        active_learning_dm = ActiveLearningDataModule(
            dm,
            initial_num_labels=initial_num_labels,
            query_size=query_size,
            val_split=0.5,
        )
        if initial_num_labels == 0:
            assert len(w) == 1
            assert "No labels provided for the initial step" in str(w[-1].message)
        else:
            assert len(w) == 0

    head = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(512, active_learning_dm.num_classes),
    )

    model = ImageClassifier(
        backbone="resnet18", head=head, num_classes=active_learning_dm.num_classes, serializer=Probabilities()
    )
    trainer = flash.Trainer(max_epochs=3)

    active_learning_loop = ActiveLearningLoop(label_epoch_frequency=1)
    active_learning_loop.connect(trainer.fit_loop)
    trainer.fit_loop = active_learning_loop

    trainer.finetune(model, datamodule=active_learning_dm, strategy="freeze")
    if initial_num_labels == 0:
        assert len(active_learning_dm._dataset) == 15
    else:
        # the last batch is 4 samples!
        assert len(active_learning_dm._dataset) == 19
    assert active_learning_loop.progress.total.completed == 3
    labelled = active_learning_loop.state_dict()["state_dict"]["datamodule_state_dict"]["labelled"]
    assert isinstance(labelled, np.ndarray)

    if initial_num_labels == 0:
        assert len(active_learning_dm.val_dataloader()) == 4
    else:
        # in the second scenario we have more labelled data!
        assert len(active_learning_dm.val_dataloader()) == 5

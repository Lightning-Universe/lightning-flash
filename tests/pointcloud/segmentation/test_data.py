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
from os.path import join

import pytest
import torch
from pytorch_lightning import seed_everything

from flash import Trainer
from flash.core.data.io.input import DataKeys
from flash.core.data.utils import download_data
from flash.core.utilities.imports import _POINTCLOUD_TESTING
from flash.pointcloud.segmentation import PointCloudSegmentation, PointCloudSegmentationData


@pytest.mark.skipif(not _POINTCLOUD_TESTING, reason="pointcloud libraries aren't installed")
def test_pointcloud_segmentation_data(tmpdir):
    seed_everything(52)

    download_data("https://pl-flash-data.s3.amazonaws.com/SemanticKittiMicro.zip", tmpdir)

    datamodule = PointCloudSegmentationData.from_folders(
        train_folder=join(tmpdir, "SemanticKittiMicro", "train"),
        predict_folder=join(tmpdir, "SemanticKittiMicro", "predict"),
        batch_size=4,
    )

    class MockModel(PointCloudSegmentation):
        def training_step(self, batch, batch_idx: int):
            assert batch[DataKeys.INPUT]["xyz"][0].shape == torch.Size([2, 45056, 3])
            assert batch[DataKeys.INPUT]["xyz"][1].shape == torch.Size([2, 11264, 3])
            assert batch[DataKeys.INPUT]["xyz"][2].shape == torch.Size([2, 2816, 3])
            assert batch[DataKeys.INPUT]["xyz"][3].shape == torch.Size([2, 704, 3])
            assert batch[DataKeys.INPUT]["labels"].shape == torch.Size([2, 45056])
            assert batch[DataKeys.INPUT]["labels"].max() == 19
            assert batch[DataKeys.INPUT]["labels"].min() == 0
            assert batch[DataKeys.METADATA][0]["name"] in ("00_000000", "00_000001")
            assert batch[DataKeys.METADATA][1]["name"] in ("00_000000", "00_000001")

    num_classes = 19
    model = MockModel(backbone="randlanet", num_classes=num_classes)
    trainer = Trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=0)
    trainer.fit(model, datamodule=datamodule)

    predictions = trainer.predict(model, datamodule=datamodule)[0]
    assert predictions[0][DataKeys.INPUT].shape == torch.Size([45056, 3])
    assert predictions[0][DataKeys.PREDS].shape == torch.Size([45056, 19])
    assert predictions[0][DataKeys.TARGET].shape == torch.Size([45056])

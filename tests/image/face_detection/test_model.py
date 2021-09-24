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
import pytest

import flash
from flash.core.utilities.imports import _FASTFACE_AVAILABLE
from flash.image import FaceDetectionData, FaceDetector

if _FASTFACE_AVAILABLE:
    import fastface as ff


@pytest.mark.skipif(not _FASTFACE_AVAILABLE, reason="vissl not installed.")
def test_fastface_training():
    dataset = ff.dataset.FDDBDataset(source_dir="data/", phase="val")
    datamodule = FaceDetectionData.from_datasets(train_dataset=dataset, batch_size=2)

    model = FaceDetector(model="lffd_slim")

    trainer = flash.Trainer(max_steps=2, num_sanity_val_steps=0)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

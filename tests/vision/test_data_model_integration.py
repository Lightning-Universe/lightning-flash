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

from flash import Trainer
from flash.vision import ImageClassificationData, ImageClassifier


def _dummy_image_loader(_):
    return torch.rand(3, 224, 224)


def test_classification(tmpdir):
    data = ImageClassificationData.from_filepaths(
        train_filepaths=["a", "b"],
        train_labels=[0, 1],
        train_transform=lambda x: x,
        loader=_dummy_image_loader,
        num_workers=0,
        batch_size=2,
    )
    model = ImageClassifier(2, backbone="resnet18")
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.finetune(model, datamodule=data, strategy="freeze")

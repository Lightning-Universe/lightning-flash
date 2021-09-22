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
import re

import pytest
import torch

import flash
from flash.core.utilities.imports import _IMAGE_AVAILABLE, _TORCHVISION_AVAILABLE, _VISSL_AVAILABLE
from flash.image import ImageClassificationData, ImageEmbedder

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import FakeData
else:
    FakeData = object

# TODO: Figure out why VISSL can't be jitted
# @pytest.mark.skipif(not (_TORCHVISION_AVAILABLE and _VISSL_AVAILABLE), reason="vissl not installed.")
# @pytest.mark.parametrize("jitter, args", [(torch.jit.trace, (torch.rand(1, 3, 64, 64),))])
# def test_jit(tmpdir, jitter, args):
#     path = os.path.join(tmpdir, "test.pt")
#
#     model = ImageEmbedder(training_strategy="barlow_twins")
#     model.eval()
#
#     model = jitter(model, *args)
#
#     torch.jit.save(model, path)
#     model = torch.jit.load(path)
#
#     out = model(torch.rand(1, 3, 64, 64))
#     assert isinstance(out, torch.Tensor)
#     assert out.shape == torch.Size([1, 2048])


@pytest.mark.skipif(_IMAGE_AVAILABLE, reason="image libraries are installed.")
def test_load_from_checkpoint_dependency_error():
    with pytest.raises(ModuleNotFoundError, match=re.escape("'lightning-flash[image]'")):
        ImageEmbedder.load_from_checkpoint("not_a_real_checkpoint.pt")


@pytest.mark.skipif(not (_TORCHVISION_AVAILABLE and _VISSL_AVAILABLE), reason="vissl not installed.")
@pytest.mark.parametrize("backbone, training_strategy", [("resnet", "barlow_twins")])
def test_vissl_training(tmpdir, backbone, training_strategy):
    datamodule = ImageClassificationData.from_datasets(
        train_dataset=FakeData(),
        batch_size=4,
    )

    embedder = ImageEmbedder(
        backbone=backbone,
        training_strategy=training_strategy,
        head="simclr_head",
        pretraining_transform="barlow_twins_transform",
        training_strategy_kwargs={"latent_embedding_dim": 128},
        pretraining_transform_kwargs={
            "total_num_crops": 2,
            "num_crops": [2],
            "size_crops": [96],
            "crop_scales": [[0.4, 1]],
        },
    )

    trainer = flash.Trainer(max_steps=3, max_epochs=1, gpus=torch.cuda.device_count())
    trainer.fit(embedder, datamodule=datamodule)

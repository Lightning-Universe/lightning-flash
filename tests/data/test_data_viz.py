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

import kornia as K
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import seed_everything

from flash.data.utils import _STAGES_PREFIX
from flash.vision import ImageClassificationData


def _rand_image():
    return Image.fromarray(np.random.randint(0, 255, (196, 196, 3), dtype="uint8"))


def test_base_viz(tmpdir):
    seed_everything(42)
    tmpdir = Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "b").mkdir()
    _rand_image().save(tmpdir / "a" / "a_1.png")
    _rand_image().save(tmpdir / "a" / "a_2.png")

    _rand_image().save(tmpdir / "b" / "a_1.png")
    _rand_image().save(tmpdir / "b" / "a_2.png")

    img_data = ImageClassificationData.from_filepaths(
        train_filepaths=[tmpdir / "a", tmpdir / "b"],
        train_labels=[0, 1],
        val_filepaths=[tmpdir / "a", tmpdir / "b"],
        val_labels=[0, 1],
        test_filepaths=[tmpdir / "a", tmpdir / "b"],
        test_labels=[0, 1],
        predict_filepaths=[tmpdir / "a", tmpdir / "b"],
        batch_size=2,
        num_workers=0,
    )

    for stage in _STAGES_PREFIX.values():

        getattr(img_data, f"show_{stage}_batch")(reset=False)
        is_predict = stage == "predict"

        def extract_data(data):
            if not is_predict:
                return data[0][0]
            return data[0]

        assert isinstance(extract_data(img_data.viz.batches[stage]["load_sample"]), Image.Image)
        if not is_predict:
            assert isinstance(img_data.viz.batches[stage]["load_sample"][0][1], int)

        assert isinstance(extract_data(img_data.viz.batches[stage]["to_tensor_transform"]), torch.Tensor)
        if not is_predict:
            assert isinstance(img_data.viz.batches[stage]["to_tensor_transform"][0][1], int)

        assert extract_data(img_data.viz.batches[stage]["collate"]).shape == torch.Size([2, 3, 196, 196])
        if not is_predict:
            assert img_data.viz.batches[stage]["collate"][0][1].shape == torch.Size([2])

        generated = extract_data(img_data.viz.batches[stage]["per_batch_transform"]).shape
        assert generated == torch.Size([2, 3, 196, 196])
        if not is_predict:
            assert img_data.viz.batches[stage]["per_batch_transform"][0][1].shape == torch.Size([2])

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
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from flash.core import Task
from flash.data.auto_dataset import AutoDataset
from flash.data.base_viz import BaseViz
from flash.data.batch import _PostProcessor, _PreProcessor
from flash.data.data_module import DataModule
from flash.data.data_pipeline import _StageOrchestrator, DataPipeline
from flash.data.process import Postprocess, Preprocess
from flash.vision import ImageClassificationData


def _rand_image():
    return Image.fromarray(np.random.randint(0, 255, (196, 196, 3), dtype="uint8"))


class ImageClassificationDataViz(ImageClassificationData):

    def configure_vis(self):
        if not hasattr(self, "viz"):
            return BaseViz(self)
        return self.viz

    def show_train_batch(self):
        self.viz = self.configure_vis()
        _ = next(iter(self.train_dataloader()))


def test_base_viz(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "b").mkdir()
    _rand_image().save(tmpdir / "a" / "a_1.png")
    _rand_image().save(tmpdir / "a" / "a_2.png")

    _rand_image().save(tmpdir / "b" / "a_1.png")
    _rand_image().save(tmpdir / "b" / "a_2.png")

    img_data = ImageClassificationDataViz.from_filepaths(
        train_filepaths=[tmpdir / "a", tmpdir / "b"],
        train_transform=None,
        train_labels=[0, 1],
        batch_size=1,
        num_workers=0,
    )

    img_data.show_train_batch()
    assert img_data.viz.batches["train"]["load_sample"] is not None
    assert img_data.viz.batches["train"]["to_tensor_transform"] is not None
    assert img_data.viz.batches["train"]["collate"] is not None
    assert img_data.viz.batches["train"]["per_batch_transform"] is not None

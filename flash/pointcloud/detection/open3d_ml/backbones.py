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
import os
from abc import ABC
from typing import Callable

import torch
from pytorch_lightning.utilities.cloud_io import load as pl_load

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _POINTCLOUD_AVAILABLE
from flash.core.utilities.providers import _OPEN3D_ML
from flash.core.utilities.url_error import catch_url_error

ROOT_URL = "https://storage.googleapis.com/open3d-releases/model-zoo/"

if _POINTCLOUD_AVAILABLE:
    import open3d
    import open3d.ml as _ml3d
    from open3d._ml3d.torch.dataloaders.concat_batcher import ConcatBatcher, ObjectDetectBatch
    from open3d._ml3d.torch.models.point_pillars import PointPillars
    from open3d.ml.torch.dataloaders import DefaultBatcher
else:
    ObjectDetectBatch = ABC
    PointPillars = ABC


class ObjectDetectBatchCollator(ObjectDetectBatch):
    def __init__(self, batches):
        self.num_batches = len(batches)
        super().__init__(batches)

    def to(self, device):
        super().to(device)
        return self

    def __len__(self):
        return self.num_batches


def register_open_3d_ml(register: FlashRegistry):

    if _POINTCLOUD_AVAILABLE:

        CONFIG_PATH = os.path.join(os.path.dirname(open3d.__file__), "_ml3d/configs")

        def get_collate_fn(model) -> Callable:
            batcher_name = model.cfg.batcher
            if batcher_name == "DefaultBatcher":
                batcher = DefaultBatcher()
            elif batcher_name == "ConcatBatcher":
                batcher = ConcatBatcher(torch, model.__class__.__name__)
            elif batcher_name == "ObjectDetectBatchCollator":
                return ObjectDetectBatchCollator
            return batcher.collate_fn

        @register(parameters=PointPillars.__init__, providers=_OPEN3D_ML)
        @catch_url_error
        def pointpillars_kitti(*args, pretrained: bool = True, **kwargs) -> PointPillars:
            cfg = _ml3d.utils.Config.load_from_file(os.path.join(CONFIG_PATH, "pointpillars_kitti.yml"))
            cfg.model.device = "cpu"
            model = PointPillars(**cfg.model)
            if pretrained:
                weight_url = os.path.join(ROOT_URL, "pointpillars_kitti_202012221652utc.pth")
                model.load_state_dict(
                    pl_load(weight_url, map_location="cpu")["model_state_dict"],
                )
            model.cfg.batcher = "ObjectDetectBatchCollator"
            return model, 384, get_collate_fn(model)

        @register(parameters=PointPillars.__init__, providers=_OPEN3D_ML)
        def pointpillars(*args, **kwargs) -> PointPillars:
            model = PointPillars(*args, **kwargs)
            model.cfg.batcher = "ObjectDetectBatch"
            return model, get_collate_fn(model)

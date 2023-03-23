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
from typing import Callable

import torch
from pytorch_lightning.utilities.cloud_io import load as pl_load

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _POINTCLOUD_AVAILABLE
from flash.core.utilities.providers import _OPEN3D_ML
from flash.core.utilities.url_error import catch_url_error

ROOT_URL = "https://storage.googleapis.com/open3d-releases/model-zoo/"


def register_open_3d_ml(register: FlashRegistry):
    if _POINTCLOUD_AVAILABLE:
        import open3d
        import open3d.ml as _ml3d
        from open3d._ml3d.torch.dataloaders import ConcatBatcher, DefaultBatcher
        from open3d._ml3d.torch.models import RandLANet

        CONFIG_PATH = os.path.join(os.path.dirname(open3d.__file__), "_ml3d/configs")

        def get_collate_fn(model) -> Callable:
            batcher_name = model.cfg.batcher
            if batcher_name == "DefaultBatcher":
                batcher = DefaultBatcher()
            elif batcher_name == "ConcatBatcher":
                batcher = ConcatBatcher(torch, model.__class__.__name__)
            else:
                batcher = None
            return batcher.collate_fn

        @register(providers=_OPEN3D_ML)
        @catch_url_error
        def randlanet_s3dis(*args, use_fold_5: bool = True, pretrained: bool = True, **kwargs) -> RandLANet:
            cfg = _ml3d.utils.Config.load_from_file(os.path.join(CONFIG_PATH, "randlanet_s3dis.yml"))
            model = RandLANet(**cfg.model)
            if pretrained:
                if use_fold_5:
                    weight_url = os.path.join(ROOT_URL, "randlanet_s3dis_area5_202010091333utc.pth")
                else:
                    weight_url = os.path.join(ROOT_URL, "randlanet_s3dis_202010091238.pth")
                model.load_state_dict(pl_load(weight_url, map_location="cpu")["model_state_dict"])
            return model, 32, get_collate_fn(model)

        @register(providers=_OPEN3D_ML)
        @catch_url_error
        def randlanet_toronto3d(*args, pretrained: bool = True, **kwargs) -> RandLANet:
            cfg = _ml3d.utils.Config.load_from_file(os.path.join(CONFIG_PATH, "randlanet_toronto3d.yml"))
            model = RandLANet(**cfg.model)
            if pretrained:
                model.load_state_dict(
                    pl_load(os.path.join(ROOT_URL, "randlanet_toronto3d_202010091306utc.pth"), map_location="cpu")[
                        "model_state_dict"
                    ],
                )
            return model, 32, get_collate_fn(model)

        @register(providers=_OPEN3D_ML)
        @catch_url_error
        def randlanet_semantic_kitti(*args, pretrained: bool = True, **kwargs) -> RandLANet:
            cfg = _ml3d.utils.Config.load_from_file(os.path.join(CONFIG_PATH, "randlanet_semantickitti.yml"))
            model = RandLANet(**cfg.model)
            if pretrained:
                model.load_state_dict(
                    pl_load(os.path.join(ROOT_URL, "randlanet_semantickitti_202009090354utc.pth"), map_location="cpu")[
                        "model_state_dict"
                    ],
                )
            return model, 32, get_collate_fn(model)

        @register(providers=_OPEN3D_ML)
        def randlanet(*args, **kwargs) -> RandLANet:
            model = RandLANet(*args, **kwargs)
            return model, 32, get_collate_fn(model)

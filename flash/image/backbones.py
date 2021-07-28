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
from torch import nn
from torch.hub import load_state_dict_from_url

from functools import partial
from typing import Tuple, Union

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone




if _TORCHVISION_AVAILABLE:

    RESNET50_WEIGHTS_PATHS = {
        "supervised": None,
        "simclr": HTTPS_VISSL + "simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/"
        "model_final_checkpoint_phase799.torch",
        "swav": HTTPS_VISSL + "swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676/"
        "model_final_checkpoint_phase799.torch",
        "barlow-twins": HTTPS_VISSL + "barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch",
    }

    def _fn_resnet(model_name: str,
                   pretrained: Union[bool, str] = True,
                   weights_paths: dict = {"supervised": None}) -> Tuple[nn.Module, int]:
        # load according to pretrained if a bool is specified, else set to False
        pretrained_flag = (pretrained and isinstance(pretrained, bool)) or (pretrained == "supervised")

        model: nn.Module = getattr(torchvision.models, model_name, None)(pretrained_flag)
        backbone = nn.Sequential(*list(model.children())[:-2])
        num_features = model.fc.in_features

        model_weights = None
        if not pretrained_flag and isinstance(pretrained, str):
            if pretrained in weights_paths:
                device = next(model.parameters()).get_device()
                model_weights = load_state_dict_from_url(
                    weights_paths[pretrained],
                    map_location=torch.device('cpu') if device == -1 else torch.device(device)
                )

                # add logic here for loading resnet weights from other libraries
                if "classy_state_dict" in model_weights.keys():
                    model_weights = model_weights["classy_state_dict"]["base_model"]["model"]["trunk"]
                    model_weights = {
                        key.replace("_feature_blocks.", "") if "_feature_blocks." in key else key: val
                        for (key, val) in model_weights.items()
                    }
                else:
                    raise KeyError('Unrecognized state dict. Logic for loading the current state dict missing.')
            else:
                raise KeyError(
                    "Requested weights for {0} not available,"
                    " choose from one of {1}".format(model_name, list(weights_paths.keys()))
                )

        if model_weights is not None:
            model.load_state_dict(model_weights, strict=False)

        return backbone, num_features



    for model_name in RESNET_MODELS:
        clf_kwargs = dict(
            fn=catch_url_error(partial(_fn_resnet, model_name=model_name)),
            name=model_name,
            namespace="vision",
            package="torchvision",
            type="resnet",
            weights_paths={"supervised": None}
        )

        if model_name == 'resnet50':
            clf_kwargs.update(
                dict(
                    fn=catch_url_error(
                        partial(_fn_resnet, model_name=model_name, weights_paths=RESNET50_WEIGHTS_PATHS)
                    ),
                    package="multiple",
                    weights_paths=RESNET50_WEIGHTS_PATHS
                )
            )
        IMAGE_CLASSIFIER_BACKBONES(**clf_kwargs)








# Wide resnet models from self-supervised training pipelines
RESNET50_W2_WEIGHTS = {
    'swav': None,
    'simclr': HTTPS_VISSL + 'simclr_rn50w2_1000ep_simclr_8node_resnet_16_07_20.e1e3bbf0/'
        'model_final_checkpoint_phase999.torch',
}

@IMAGE_CLASSIFIER_BACKBONES(name='resnet50-w2', namespace="vision", package="multiple",
                            type="resnet", weights_paths=RESNET50_W2_WEIGHTS)
def _wide_resnet50_w2(pretrained: str, weights_paths: dict = RESNET50_W2_WEIGHTS) -> Tuple[nn.Module, int]:
    if not type(pretrained) == str:
        raise TypeError('pretrained param should be str.')

    if pretrained not in weights_paths:
        raise KeyError(
            "Requested weights for Resnet50-w2 not available,"
            " choose from one of {0}".format(list(weights_paths.keys()))
        )

    backbone = torch.hub.load('facebookresearch/swav', 'resnet50w2')

    model_weights = None
    if pretrained != 'swav':
        device = next(backbone.parameters()).get_device()
        model_weights = load_state_dict_from_url(
            weights_paths[pretrained],
            map_location=torch.device('cpu') if device == -1 else torch.device(device)
        )

    return backbone, 4096


RESNET50_W4_WEIGHTS = {
    'swav': None,
    'simclr': HTTPS_VISSL + 'simclr_rn50w4_1000ep_bs32_16node_simclr_8node_resnet_28_07_20.9e20b0ae/'
        'model_final_checkpoint_phase999.torch',
}

@IMAGE_CLASSIFIER_BACKBONES(name='resnet50-w4', namespace="vision", package="multiple",
                            type="resnet", weights_paths=RESNET50_W4_WEIGHTS)
def _wide_resnet50_w4(pretrained: str, weights_paths: dict = RESNET50_W4_WEIGHTS) -> Tuple[nn.Module, int]:
    backbone = torch.hub.load('facebookresearch/swav', 'resnet50w4')

    return backbone, 8192




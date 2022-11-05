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
#
#
# ResNet encoder adapted from: https://github.com/facebookresearch/swav/blob/master/src/resnet50.py
# as the official torchvision implementation does not support wide resnet architecture
# found in self-supervised learning model weights
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TIMM_AVAILABLE
from flash.core.utilities.url_error import catch_url_error

if _TIMM_AVAILABLE:
    from timm.models.helpers import adapt_input_conv


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        widen: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        first_conv3x3: bool = False,
        remove_first_maxpool: bool = False,
        in_chans: int = 3,
    ) -> None:

        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        num_out_filters = width_per_group * widen

        if first_conv3x3:
            self.conv1 = nn.Conv2d(in_chans, num_out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_chans, num_out_filters, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)

        if remove_first_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def _resnet(
    model_name: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    num_features: int,
    pretrained: Union[bool, str] = True,
    weights_paths: dict = {"supervised": None},
    **kwargs: Any,
) -> ResNet:

    pretrained_flag = (pretrained and isinstance(pretrained, bool)) or (pretrained == "supervised")

    backbone = ResNet(block, layers, **kwargs)
    device = next(backbone.parameters()).get_device()

    model_weights = None
    if pretrained_flag:
        if "supervised" not in weights_paths:
            raise KeyError(f"Supervised pretrained weights not available for {model_name}")

        model_weights = load_state_dict_from_url(
            weights_paths["supervised"], map_location=torch.device("cpu") if device == -1 else torch.device(device)
        )

        # for supervised pretrained weights
        model_weights.pop("fc.weight")
        model_weights.pop("fc.bias")

    if not pretrained_flag and isinstance(pretrained, str):
        if pretrained in weights_paths:
            model_weights = load_state_dict_from_url(
                weights_paths[pretrained], map_location=torch.device("cpu") if device == -1 else torch.device(device)
            )

            if "classy_state_dict" in model_weights.keys():
                model_weights = model_weights["classy_state_dict"]["base_model"]["model"]["trunk"]
                model_weights = {
                    key.replace("_feature_blocks.", "") if "_feature_blocks." in key else key: val
                    for (key, val) in model_weights.items()
                }
            else:
                raise KeyError("Unrecognized state dict. Logic for loading the current state dict missing.")
        else:
            raise KeyError(
                f"Requested weights for {model_name} not available," f" choose from one of {weights_paths.keys()}"
            )

    if model_weights is not None:
        in_chans = backbone.conv1.weight.shape[1]
        model_weights["conv1.weight"] = adapt_input_conv(in_chans, model_weights["conv1.weight"])
        backbone.load_state_dict(model_weights)

    return backbone, num_features


HTTPS_VISSL = "https://dl.fbaipublicfiles.com/vissl/model_zoo/"
RESNET50_WEIGHTS_PATHS = {
    "supervised": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "simclr": HTTPS_VISSL + "simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/"
    "model_final_checkpoint_phase799.torch",
    "swav": HTTPS_VISSL + "swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676/"
    "model_final_checkpoint_phase799.torch",
}
RESNET50W2_WEIGHTS_PATHS = {
    "simclr": HTTPS_VISSL + "simclr_rn50w2_1000ep_simclr_8node_resnet_16_07_20.e1e3bbf0/"
    "model_final_checkpoint_phase999.torch",
    "swav": HTTPS_VISSL + "swav_rn50w2_in1k_bs32_16node_400ep_swav_8node_resnet_30_07_20.93563e51/"
    "model_final_checkpoint_phase399.torch",
}
RESNET50W4_WEIGHTS_PATHS = {
    "simclr": HTTPS_VISSL + "simclr_rn50w4_1000ep_bs32_16node_simclr_8node_resnet_28_07_20.9e20b0ae/"
    "model_final_checkpoint_phase999.torch",
    "swav": HTTPS_VISSL + "swav_rn50w4_in1k_bs40_8node_400ep_swav_8node_resnet_30_07_20.1736135b/"
    "model_final_checkpoint_phase399.torch",
}

RESNET_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet50w2", "resnet50w4"]
RESNET_PARAMS = [
    {
        "block": BasicBlock,
        "layers": [2, 2, 2, 2],
        "num_features": 512,
        "weights_paths": {"supervised": "https://download.pytorch.org/models/resnet18-f37072fd.pth"},
    },
    {
        "block": BasicBlock,
        "layers": [3, 4, 6, 3],
        "num_features": 512,
        "weights_paths": {"supervised": "https://download.pytorch.org/models/resnet34-b627a593.pth"},
    },
    {"block": Bottleneck, "layers": [3, 4, 6, 3], "num_features": 2048, "weights_paths": RESNET50_WEIGHTS_PATHS},
    {
        "block": Bottleneck,
        "layers": [3, 4, 23, 3],
        "num_features": 2048,
        "weights_paths": {"supervised": "https://download.pytorch.org/models/resnet101-63fe2227.pth"},
    },
    {
        "block": Bottleneck,
        "layers": [3, 8, 36, 3],
        "num_features": 2048,
        "weights_paths": {"supervised": "https://download.pytorch.org/models/resnet152-394f9c45.pth"},
    },
    {
        "block": Bottleneck,
        "layers": [3, 4, 6, 3],
        "widen": 2,
        "num_features": 4096,
        "weights_paths": RESNET50W2_WEIGHTS_PATHS,
    },
    {
        "block": Bottleneck,
        "layers": [3, 4, 6, 3],
        "widen": 4,
        "num_features": 8192,
        "weights_paths": RESNET50W4_WEIGHTS_PATHS,
    },
]


def register_resnet_backbones(register: FlashRegistry):
    for model_name, params in zip(RESNET_MODELS, RESNET_PARAMS):
        register(
            fn=catch_url_error(partial(_resnet, model_name=model_name, **params)),
            name=model_name,
            namespace="vision",
            package="multiple",
            type="resnet",
            weights_paths=params["weights_paths"],  # update
        )

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
from dataclasses import dataclass
from typing import Callable

import torch

from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _KORNIA_AVAILABLE, _PYTORCHVIDEO_AVAILABLE, requires

if _KORNIA_AVAILABLE:
    import kornia.augmentation as K

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.transforms import UniformTemporalSubsample
    from torchvision.transforms import CenterCrop, Compose, RandomCrop
else:
    ClipSampler, LabeledVideoDataset, EncodedVideo, ApplyTransformToKey = None, None, None, None


@requires("video")
@dataclass
class VideoClassificationInputTransform(InputTransform):

    image_size: int = 244
    temporal_sub_sample: int = 8
    mean: torch.Tensor = torch.tensor([0.45, 0.45, 0.45])
    std: torch.Tensor = torch.tensor([0.225, 0.225, 0.225])
    data_format: str = "BCTHW"
    same_on_frame: bool = False

    def per_sample_transform(self) -> Callable:
        if self.training:
            per_sample_transform = [RandomCrop(self.image_size, pad_if_needed=True)]
        else:
            per_sample_transform = [CenterCrop(self.image_size)]

        return ApplyToKeys(
            "video", Compose([UniformTemporalSubsample(self.temporal_sub_sample)] + per_sample_transform)
        )

    def per_batch_transform_on_device(self) -> Callable:
        return ApplyToKeys(
            "video",
            K.VideoSequential(
                K.Normalize(self.mean, self.std),
                data_format=self.data_format,
                same_on_frame=self.same_on_frame,
            ),
        )

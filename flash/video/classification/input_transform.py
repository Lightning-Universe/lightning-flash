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
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import Sampler

from flash.core.data.io.input import InputFormat
from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.imports import _KORNIA_AVAILABLE, _PYTORCHVIDEO_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as K

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.data.clip_sampling import ClipSampler, make_clip_sampler
    from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
    from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip
else:
    ClipSampler, LabeledVideoDataset, EncodedVideo, ApplyTransformToKey = None, None, None, None

_PYTORCHVIDEO_DATA = Dict[str, Union[str, torch.Tensor, int, float, List]]

Label = Union[int, List[int]]


class VideoClassificationInputTransform(InputTransform):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        **_kwargs: Any,
    ):
        self.clip_sampler = clip_sampler
        self.clip_duration = clip_duration
        self.clip_sampler_kwargs = clip_sampler_kwargs
        self.video_sampler = video_sampler
        self.decode_audio = decode_audio
        self.decoder = decoder

        if not _PYTORCHVIDEO_AVAILABLE:
            raise ModuleNotFoundError("Please, run `pip install pytorchvideo`.")

        if not clip_sampler_kwargs:
            clip_sampler_kwargs = {}

        if not clip_sampler:
            raise MisconfigurationException(
                "clip_sampler should be provided as a string or ``pytorchvideo.data.clip_sampling.ClipSampler``"
            )

        clip_sampler = make_clip_sampler(clip_sampler, clip_duration, **clip_sampler_kwargs)

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.FILES: VideoClassificationPathsPredictInput,
                InputFormat.FOLDERS: VideoClassificationPathsPredictInput,
                InputFormat.FIFTYONE: VideoClassificationFiftyOneInput,
            },
            default_input=InputFormat.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "clip_sampler": self.clip_sampler,
            "clip_duration": self.clip_duration,
            "clip_sampler_kwargs": self.clip_sampler_kwargs,
            "video_sampler": self.video_sampler,
            "decode_audio": self.decode_audio,
            "decoder": self.decoder,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool) -> "VideoClassificationInputTransform":
        return cls(**state_dict)

    def default_transforms(self) -> Dict[str, Callable]:
        if self.training:
            per_sample_transform = [
                RandomCrop(244, pad_if_needed=True),
                RandomHorizontalFlip(p=0.5),
            ]
        else:
            per_sample_transform = [
                CenterCrop(244),
            ]

        return {
            "per_sample_transform": Compose(
                [
                    ApplyTransformToKey(
                        key="video",
                        transform=Compose([UniformTemporalSubsample(8)] + per_sample_transform),
                    ),
                ]
            ),
            "per_batch_transform_on_device": Compose(
                [
                    ApplyTransformToKey(
                        key="video",
                        transform=K.VideoSequential(
                            K.Normalize(torch.tensor([0.45, 0.45, 0.45]), torch.tensor([0.225, 0.225, 0.225])),
                            data_format="BCTHW",
                            same_on_frame=False,
                        ),
                    ),
                ]
            ),
        }


@dataclass
class VideoClassificationInputTransform(InputTransform):

    image_size: int = 244
    clip_sampler: Union[str, "ClipSampler"] = "random"
    clip_duration: float = 2
    clip_sampler_kwargs: Dict[str, Any] = None
    video_sampler: Type[Sampler] = torch.utils.data.RandomSampler
    decode_audio: bool = False
    decoder: str = "pyav"
    temporal_sub_sample: int = 8

    def input_per_sample_transform(self):
        if self.training:
            per_sample_transform = [
                RandomCrop(self.image_size, pad_if_needed=True),
                RandomHorizontalFlip(p=0.5),
            ]
        else:
            per_sample_transform = [
                CenterCrop(self.image_size),
            ]

        return ApplyTransformToKey(
            key="video",
            transform=Compose([UniformTemporalSubsample(8)] + per_sample_transform),
        )

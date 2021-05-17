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
import pathlib
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import RandomSampler, Sampler

from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DefaultDataKeys, DefaultDataSources, LabelsState, PathsDataSource
from flash.core.data.process import Preprocess
from flash.core.data.transforms import merge_transforms
from flash.core.utilities.imports import _KORNIA_AVAILABLE, _PYTORCHVIDEO_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as K

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.data.clip_sampling import ClipSampler, make_clip_sampler
    from pytorchvideo.data.encoded_video import EncodedVideo
    from pytorchvideo.data.encoded_video_dataset import EncodedVideoDataset, labeled_encoded_video_dataset
    from pytorchvideo.transforms import (
        ApplyTransformToKey,
        RandomShortSideScale,
        ShortSideScale,
        UniformTemporalSubsample,
    )
    from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip
else:
    ClipSampler, EncodedVideoDataset, EncodedVideo, ApplyTransformToKey = None, None, None, None

_PYTORCHVIDEO_DATA = Dict[str, Union[str, torch.Tensor, int, float, List]]


class VideoClassificationPathsDataSource(PathsDataSource):

    def __init__(
        self,
        clip_sampler: 'ClipSampler',
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ):
        super().__init__(extensions=("mp4", "avi"))
        self.clip_sampler = clip_sampler
        self.video_sampler = video_sampler
        self.decode_audio = decode_audio
        self.decoder = decoder

    def load_data(self, data: str, dataset: Optional[Any] = None) -> 'EncodedVideoDataset':
        ds: EncodedVideoDataset = labeled_encoded_video_dataset(
            pathlib.Path(data),
            self.clip_sampler,
            video_sampler=self.video_sampler,
            decode_audio=self.decode_audio,
            decoder=self.decoder,
        )
        if self.training:
            label_to_class_mapping = {p[1]: p[0].split("/")[-2] for p in ds._labeled_videos._paths_and_labels}
            self.set_state(LabelsState(label_to_class_mapping))
            dataset.num_classes = len(np.unique([s[1]['label'] for s in ds._labeled_videos]))
        return ds

    def _encoded_video_to_dict(self, video) -> Dict[str, Any]:
        (
            clip_start,
            clip_end,
            clip_index,
            aug_index,
            is_last_clip,
        ) = self.clip_sampler(0.0, video.duration)

        loaded_clip = video.get_clip(clip_start, clip_end)

        clip_is_null = (
            loaded_clip is None or loaded_clip["video"] is None or (loaded_clip["audio"] is None and self.decode_audio)
        )

        if clip_is_null:
            raise MisconfigurationException(
                f"The provided video is too short {video.duration} to be clipped at {self.clip_sampler._clip_duration}"
            )

        frames = loaded_clip["video"]
        audio_samples = loaded_clip["audio"]
        return {
            "video": frames,
            "video_name": video.name,
            "video_index": 0,
            "clip_index": clip_index,
            "aug_index": aug_index,
            **({
                "audio": audio_samples
            } if audio_samples is not None else {}),
        }

    def predict_load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self._encoded_video_to_dict(EncodedVideo.from_path(sample[DefaultDataKeys.INPUT]))


class VideoClassificationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        clip_sampler: Union[str, 'ClipSampler'] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = True,
        decoder: str = "pyav",
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
            data_sources={
                DefaultDataSources.FILES: VideoClassificationPathsDataSource(
                    clip_sampler,
                    video_sampler=video_sampler,
                    decode_audio=decode_audio,
                    decoder=decoder,
                ),
                DefaultDataSources.FOLDERS: VideoClassificationPathsDataSource(
                    clip_sampler,
                    video_sampler=video_sampler,
                    decode_audio=decode_audio,
                    decoder=decoder,
                ),
            },
            default_data_source=DefaultDataSources.FILES,
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
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool) -> 'VideoClassificationPreprocess':
        return cls(**state_dict)

    def default_transforms(self) -> Dict[str, Callable]:
        if self.training:
            post_tensor_transform = [
                RandomShortSideScale(min_size=256, max_size=320),
                RandomCrop(244),
                RandomHorizontalFlip(p=0.5),
            ]
        else:
            post_tensor_transform = [
                ShortSideScale(256),
            ]

        return {
            "post_tensor_transform": Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose([UniformTemporalSubsample(8)] + post_tensor_transform),
                ),
            ]),
            "per_batch_transform_on_device": Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=K.VideoSequential(
                        K.Normalize(torch.tensor([0.45, 0.45, 0.45]), torch.tensor([0.225, 0.225, 0.225])),
                        data_format="BCTHW",
                        same_on_frame=False
                    )
                ),
            ]),
        }


class VideoClassificationData(DataModule):
    """Data module for Video classification tasks."""

    preprocess_cls = VideoClassificationPreprocess

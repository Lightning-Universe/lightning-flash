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
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING, Union

import numpy as np
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import Sampler

from flash.core.data.data_module import DataModule
from flash.core.data.data_source import (
    DefaultDataKeys,
    DefaultDataSources,
    FiftyOneDataSource,
    LabelsState,
    PathsDataSource,
)
from flash.core.data.process import Preprocess
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _KORNIA_AVAILABLE, _PYTORCHVIDEO_AVAILABLE, lazy_import

SampleCollection = None
if _FIFTYONE_AVAILABLE:
    fol = lazy_import("fiftyone.core.labels")
    if TYPE_CHECKING:
        from fiftyone.core.collections import SampleCollection
else:
    fol = None

if _KORNIA_AVAILABLE:
    import kornia.augmentation as K

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.data.clip_sampling import ClipSampler, make_clip_sampler
    from pytorchvideo.data.encoded_video import EncodedVideo
    from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset, LabeledVideoDataset
    from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
    from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
    from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip
else:
    ClipSampler, LabeledVideoDataset, EncodedVideo, ApplyTransformToKey = None, None, None, None

_PYTORCHVIDEO_DATA = Dict[str, Union[str, torch.Tensor, int, float, List]]


class BaseVideoClassification:
    def __init__(
        self,
        clip_sampler: "ClipSampler",
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ):
        self.clip_sampler = clip_sampler
        self.video_sampler = video_sampler
        self.decode_audio = decode_audio
        self.decoder = decoder

    def load_data(self, data: str, dataset: Optional[Any] = None) -> "LabeledVideoDataset":
        ds = self._make_encoded_video_dataset(data)
        if self.training:
            label_to_class_mapping = {p[1]: p[0].split("/")[-2] for p in ds._labeled_videos._paths_and_labels}
            self.set_state(LabelsState(label_to_class_mapping))
            dataset.num_classes = len(np.unique([s[1]["label"] for s in ds._labeled_videos]))
        return ds

    def load_sample(self, sample):
        return sample

    def predict_load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        video_path = sample[DefaultDataKeys.INPUT]
        sample.update(self._encoded_video_to_dict(EncodedVideo.from_path(video_path)))
        sample[DefaultDataKeys.METADATA] = {"filepath": video_path}
        return sample

    def _encoded_video_to_dict(self, video, annotation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        (
            clip_start,
            clip_end,
            clip_index,
            aug_index,
            is_last_clip,
        ) = self.clip_sampler(0.0, video.duration, annotation)

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
            **({"audio": audio_samples} if audio_samples is not None else {}),
        }

    def _make_encoded_video_dataset(self, data) -> "LabeledVideoDataset":
        raise NotImplementedError("Subclass must implement _make_encoded_video_dataset()")


class VideoClassificationPathsDataSource(BaseVideoClassification, PathsDataSource):
    def __init__(
        self,
        clip_sampler: "ClipSampler",
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ):
        super().__init__(
            clip_sampler,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
        )
        PathsDataSource.__init__(
            self,
            extensions=("mp4", "avi"),
        )

    def _make_encoded_video_dataset(self, data) -> "LabeledVideoDataset":
        ds: LabeledVideoDataset = labeled_video_dataset(
            pathlib.Path(data),
            self.clip_sampler,
            video_sampler=self.video_sampler,
            decode_audio=self.decode_audio,
            decoder=self.decoder,
        )
        return ds


class VideoClassificationFiftyOneDataSource(
    BaseVideoClassification,
    FiftyOneDataSource,
):
    def __init__(
        self,
        clip_sampler: "ClipSampler",
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = True,
        decoder: str = "pyav",
        label_field: str = "ground_truth",
    ):
        super().__init__(
            clip_sampler=clip_sampler,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
        )
        FiftyOneDataSource.__init__(
            self,
            label_field=label_field,
        )

    @property
    def label_cls(self):
        return fol.Classification

    def _make_encoded_video_dataset(self, data: SampleCollection) -> "LabeledVideoDataset":
        classes = self._get_classes(data)
        label_to_class_mapping = dict(enumerate(classes))
        class_to_label_mapping = {c: lab for lab, c in label_to_class_mapping.items()}

        filepaths = data.values("filepath")
        labels = data.values(self.label_field + ".label")
        targets = [class_to_label_mapping[lab] for lab in labels]
        labeled_video_paths = LabeledVideoPaths(list(zip(filepaths, targets)))

        ds: LabeledVideoDataset = LabeledVideoDataset(
            labeled_video_paths,
            self.clip_sampler,
            video_sampler=self.video_sampler,
            decode_audio=self.decode_audio,
            decoder=self.decoder,
        )
        return ds


class VideoClassificationPreprocess(Preprocess):
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
        decode_audio: bool = True,
        decoder: str = "pyav",
        **data_source_kwargs: Any,
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
                DefaultDataSources.FIFTYONE: VideoClassificationFiftyOneDataSource(
                    clip_sampler,
                    video_sampler=video_sampler,
                    decode_audio=decode_audio,
                    decoder=decoder,
                    **data_source_kwargs,
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
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool) -> "VideoClassificationPreprocess":
        return cls(**state_dict)

    def default_transforms(self) -> Dict[str, Callable]:
        if self.training:
            post_tensor_transform = [
                RandomCrop(244, pad_if_needed=True),
                RandomHorizontalFlip(p=0.5),
            ]
        else:
            post_tensor_transform = [
                CenterCrop(244),
            ]

        return {
            "post_tensor_transform": Compose(
                [
                    ApplyTransformToKey(
                        key="video",
                        transform=Compose([UniformTemporalSubsample(8)] + post_tensor_transform),
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


class VideoClassificationData(DataModule):
    """Data module for Video classification tasks."""

    preprocess_cls = VideoClassificationPreprocess

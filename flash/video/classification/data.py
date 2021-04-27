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
import pathlib
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import RandomSampler, Sampler
from torch.utils.data.dataset import IterableDataset

from flash.core.classification import ClassificationState
from flash.data.data_module import DataModule
from flash.data.process import Preprocess
from flash.utils.imports import _KORNIA_AVAILABLE, _PYTORCHVIDEO_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as K
    import kornia.geometry.transform as T
else:
    from torchvision import transforms as T
if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.data.clip_sampling import ClipSampler, make_clip_sampler
    from pytorchvideo.data.encoded_video import EncodedVideo
    from pytorchvideo.data.encoded_video_dataset import EncodedVideoDataset, labeled_encoded_video_dataset
    from pytorchvideo.transforms import ApplyTransformToKey, RandomShortSideScale, UniformTemporalSubsample
    from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip
else:
    ClipSampler, EncodedVideoDataset, EncodedVideo, ApplyTransformToKey = None, None, None, None

_PYTORCHVIDEO_DATA = Dict[str, Union[str, torch.Tensor, int, float, List]]


class VideoClassificationPreprocess(Preprocess):

    EXTENSIONS = ("mp4", "avi")

    @staticmethod
    def default_predict_transform() -> Dict[str, 'Compose']:
        return {
            "post_tensor_transform": Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose([
                        UniformTemporalSubsample(8),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(244),
                        RandomHorizontalFlip(p=0.5),
                    ]),
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

    def __init__(
        self,
        clip_sampler: 'ClipSampler',
        video_sampler: Type[Sampler],
        decode_audio: bool,
        decoder: str,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
    ):
        # Make sure to provide your transform to the Preprocess Class
        super().__init__(
            train_transform, val_transform, test_transform, predict_transform or self.default_predict_transform()
        )
        self.clip_sampler = clip_sampler
        self.video_sampler = video_sampler
        self.decode_audio = decode_audio
        self.decoder = decoder

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            'clip_sampler': self.clip_sampler,
            'video_sampler': self.video_sampler,
            'decode_audio': self.decode_audio,
            'decoder': self.decoder,
            'train_transform': self._train_transform,
            'val_transform': self._val_transform,
            'test_transform': self._test_transform,
            'predict_transform': self._predict_transform,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool) -> 'VideoClassificationPreprocess':
        return cls(**state_dict)

    def load_data(self, data: Any, dataset: IterableDataset) -> 'EncodedVideoDataset':
        ds: EncodedVideoDataset = labeled_encoded_video_dataset(
            data,
            self.clip_sampler,
            video_sampler=self.video_sampler,
            decode_audio=self.decode_audio,
            decoder=self.decoder,
        )
        if self.training:
            label_to_class_mapping = {p[1]: p[0].split("/")[-2] for p in ds._labeled_videos._paths_and_labels}
            self.set_state(ClassificationState(label_to_class_mapping))
            dataset.num_classes = len(np.unique([s[1]['label'] for s in ds._labeled_videos]))
        return ds

    def predict_load_data(self, folder_or_file: Union[str, List[str]]) -> List[str]:
        if isinstance(folder_or_file, list) and all(os.path.exists(p) for p in folder_or_file):
            return folder_or_file
        elif os.path.isdir(folder_or_file):
            return [f for f in os.listdir(folder_or_file) if f.lower().endswith(self.EXTENSIONS)]
        elif os.path.exists(folder_or_file) and folder_or_file.lower().endswith(self.EXTENSIONS):
            return [folder_or_file]
        raise MisconfigurationException(
            f"The provided predict output should be a folder or a path. Found: {folder_or_file}"
        )

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

    def predict_load_sample(self, video_path: str) -> "EncodedVideo":
        return self._encoded_video_to_dict(EncodedVideo.from_path(video_path))

    def pre_tensor_transform(self, sample: _PYTORCHVIDEO_DATA) -> _PYTORCHVIDEO_DATA:
        return self.current_transform(sample)

    def to_tensor_transform(self, sample: _PYTORCHVIDEO_DATA) -> _PYTORCHVIDEO_DATA:
        return self.current_transform(sample)

    def post_tensor_transform(self, sample: _PYTORCHVIDEO_DATA) -> _PYTORCHVIDEO_DATA:
        return self.current_transform(sample)

    def per_batch_transform(self, sample: _PYTORCHVIDEO_DATA) -> _PYTORCHVIDEO_DATA:
        return self.current_transform(sample)

    def per_batch_transform_on_device(self, sample: _PYTORCHVIDEO_DATA) -> _PYTORCHVIDEO_DATA:
        return self.current_transform(sample)
        


class VideoClassificationData(DataModule):
    """Data module for Video classification tasks."""

    preprocess_cls = VideoClassificationPreprocess

    @classmethod
    def from_paths(
        cls,
        train_data_path: Optional[Union[str, pathlib.Path]] = None,
        val_data_path: Optional[Union[str, pathlib.Path]] = None,
        test_data_path: Optional[Union[str, pathlib.Path]] = None,
        predict_data_path: Union[str, pathlib.Path] = None,
        clip_sampler: Union[str, 'ClipSampler'] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = RandomSampler,
        decode_audio: bool = True,
        decoder: str = "pyav",
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        preprocess: Optional[Preprocess] = None,
        **kwargs,
    ) -> 'DataModule':
        """

        Creates a VideoClassificationData object from folders of videos arranged in this way: ::

            train/class_x/xxx.ext
            train/class_x/xxy.ext
            train/class_x/xxz.ext
            train/class_y/123.ext
            train/class_y/nsdf3.ext
            train/class_y/asd932_.ext

        Args:
            train_data_path: Path to training folder. Default: None.
            val_data_path: Path to validation folder. Default: None.
            test_data_path: Path to test folder. Default: None.
            predict_data_path: Path to predict folder. Default: None.
            clip_sampler: ClipSampler to be used on videos.
            clip_duration: Clip duration for the clip sampler.
            clip_sampler_kwargs: Extra ClipSampler keyword arguments.
            video_sampler: Sampler for the internal video container.
                This defines the order videos are decoded and, if necessary, the distributed split.
            decode_audio: Whether to decode the audio with the video clip.
            decoder: Defines what type of decoder used to decode a video.
            train_transform: Video clip dictionary transform to use for training set.
            val_transform:  Video clip dictionary transform to use for validation set.
            test_transform:  Video clip dictionary transform to use for test set.
            predict_transform:  Video clip dictionary transform to use for predict set.
            batch_size: Batch size for data loading.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to ``None`` which equals the number of available CPU threads.
            preprocess: VideoClassifierPreprocess to handle the data processing.

        Returns:
            VideoClassificationData: the constructed data module

        Examples:
            >>> videos = VideoClassificationData.from_paths("train/") # doctest: +SKIP

        """
        if not _PYTORCHVIDEO_AVAILABLE:
            raise ModuleNotFoundError("Please, run `pip install pytorchvideo`.")

        if not clip_sampler_kwargs:
            clip_sampler_kwargs = {}

        if not clip_sampler:
            raise MisconfigurationException(
                "clip_sampler should be provided as a string or ``pytorchvideo.data.clip_sampling.ClipSampler``"
            )

        clip_sampler = make_clip_sampler(clip_sampler, clip_duration, **clip_sampler_kwargs)

        preprocess: Preprocess = preprocess or cls.preprocess_cls(
            clip_sampler, video_sampler, decode_audio, decoder, train_transform, val_transform, test_transform,
            predict_transform
        )

        return cls.from_load_data_inputs(
            train_load_data_input=train_data_path,
            val_load_data_input=val_data_path,
            test_load_data_input=test_data_path,
            predict_load_data_input=predict_data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            use_iterable_auto_dataset=True,
            **kwargs,
        )

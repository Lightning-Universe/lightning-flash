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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset, RandomSampler, Sampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataset import IterableDataset
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS, make_dataset

from flash.data.auto_dataset import AutoDataset
from flash.data.data_module import DataModule
from flash.data.data_pipeline import DataPipeline
from flash.data.process import Preprocess
from flash.utils.imports import _KORNIA_AVAILABLE, _PYTORCH_VIDEO_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as K
    import kornia.geometry.transform as T
else:
    from torchvision import transforms as T

if _PYTORCH_VIDEO_AVAILABLE:
    from pytorchvideo.data.clip_sampling import ClipSampler, make_clip_sampler
    from pytorchvideo.data.encoded_video_dataset import labeled_encoded_video_dataset


class VideoPreprocessPreprocess(Preprocess):

    def __init__(
        self,
        clip_sampler: ClipSampler,
        train_transform: Optional[Union[Callable, Module, Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Callable, Module, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, Module, Dict[str, Callable]]] = None,
        predict_transform: Optional[Union[Callable, Module, Dict[str, Callable]]] = None,
    ):

        super().__init__()
        self.clip_sampler = clip_sampler

    def load_data(self, data: Any, dataset: IterableDataset) -> Dict:
        return labeled_encoded_video_dataset(
            data,
            self.clip_sampler,
        )


class VideoClassificationData(DataModule):
    """Data module for Video classification tasks."""

    preprocess_cls = VideoPreprocessPreprocess

    @classmethod
    def instantiate_preprocess(
        cls,
        clip_sampler: ClipSampler,
        video_sampler: Type[Sampler] = RandomSampler,
        train_transform: Dict[str, Union[nn.Module, Callable]],
        val_transform: Dict[str, Union[nn.Module, Callable]],
        test_transform: Dict[str, Union[nn.Module, Callable]],
        predict_transform: Dict[str, Union[nn.Module, Callable]],
        preprocess_cls: Type[Preprocess] = None
    ) -> Preprocess:
        """
        """
        preprocess_cls = preprocess_cls or cls.preprocess_cls
        preprocess: Preprocess = preprocess_cls(
            clip_sampler, train_transform, val_transform, test_transform, predict_transform
        )
        return preprocess

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[Union[str, pathlib.Path]] = None,
        val_folder: Optional[Union[str, pathlib.Path]] = None,
        test_folder: Optional[Union[str, pathlib.Path]] = None,
        predict_folder: Union[str, pathlib.Path] = None,
        clip_sampler: Union[str, ClipSampler] = "random",
        clip_duration: float = 2,
        video_sampler: Type[Sampler] = RandomSampler,
        clip_sampler_kwargs: Dict[str, Any] = None,
        train_transform: Optional[Union[str, Dict]] = 'default',
        val_transform: Optional[Union[str, Dict]] = 'default',
        test_transform: Optional[Union[str, Dict]] = 'default',
        predict_transform: Optional[Union[str, Dict]] = 'default',
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        preprocess_cls: Optional[Type[Preprocess]] = None,
        **kwargs,
    ) -> 'DataModule':
        """

        Creates a VideoClassificationData object from folders of images arranged in this way: ::

            train/class_x/xxx.ext
            train/class_x/xxy.ext
            train/class_x/xxz.ext
            train/class_y/123.ext
            train/class_y/nsdf3.ext
            train/class_y/asd932_.ext

        Args:
            train_folder: Path to training folder. Default: None.
            val_folder: Path to validation folder. Default: None.
            test_folder: Path to test folder. Default: None.
            predict_folder: Path to predict folder. Default: None.
            val_transform: Image transform to use for validation and test set.
            clip_sampler: ClipSampler to be used on videos.
            train_transform: Image transform to use for training set.
            val_transform: Image transform to use for validation set.
            test_transform: Image transform to use for test set.
            predict_transform: Image transform to use for predict set.
            batch_size: Batch size for data loading.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to ``None`` which equals the number of available CPU threads.

        Returns:
            VideoClassificationData: the constructed data module

        Examples:
            >>> img_data = VideoClassificationData.from_folders("train/") # doctest: +SKIP

        """
        if not clip_sampler_kwargs:
            clip_sampler_kwargs = {}

        if not clip_sampler:
            raise MisconfigurationException(
                "clip_sampler should be provided as a string or ``pytorchvideo.data.clip_sampling.ClipSampler``"
            )

        clip_sampler = make_clip_sampler(clip_sampler, clip_duration, **clip_sampler_kwargs)

        preprocess = cls.instantiate_preprocess(
            clip_sampler,
            video_sampler,
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
            preprocess_cls=preprocess_cls,
        )

        return cls.from_load_data_inputs(
            train_load_data_input=train_folder,
            val_load_data_input=val_folder,
            test_load_data_input=test_folder,
            predict_load_data_input=predict_folder,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            **kwargs,
        )

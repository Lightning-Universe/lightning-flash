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
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torchvision import transforms as T
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS, make_dataset

from flash.vision.classification.data import ImageClassificationDataPipeline, ImageClassificationData
from flash.audio.classification.utils import wav2spec
from flash.core.data.datamodule import DataModule
from flash.core.data.utils import _contains_any_tensor


def _pil_loader(path) -> Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f, Image.open(f) as img:
        return img.convert("RGB")

_default_train_transforms = T.Compose([
    # T.RandomResizedCrop(224),
    # T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_default_valid_transforms = T.Compose([
    # T.Resize(256),
    # T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# todo: torch.nn.modules.module.ModuleAttributeError: 'Resize' object has no attribute '_forward_hooks'
# Find better fix and raised issue on torchvision.
_default_valid_transforms.transforms[0]._forward_hooks = {}

class SpectrogramClassificationDataPipeline(ImageClassificationDataPipeline):

    def __init__(
        self,
        train_transform: Optional[Callable] = _default_train_transforms,
        valid_transform: Optional[Callable] = _default_valid_transforms,
        use_valid_transform: bool = True,
        loader: Callable = _pil_loader
    ):
        super().__init__(train_transform,
                       valid_transform,
                       use_valid_transform,
                       loader)

    def before_collate(self, samples: Any) -> Any:

        if _contains_any_tensor(samples):
            return samples

        if isinstance(samples, str):
            samples = [samples]
        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                if sample[:-3].lower() == 'wav': ## If filepath is a wav convert it to a png spectrogram 
                    wav2spec(sample)
                    sample = f'{sample[:-3]}png'
                output = self._loader(sample)
                transform = self._valid_transform if self._use_valid_transform else self._train_transform
                outputs.append(transform(output))
            return outputs
        raise MisconfigurationException("The samples should either be a tensor or a list of paths.")
            
    

class SpectrogramClassificationData(ImageClassificationData):
    @classmethod
    def from_filepaths(
        cls,
        train_filepaths: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        train_labels: Optional[Sequence] = None,
        train_transform: Optional[Callable] = _default_train_transforms,
        valid_filepaths: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        valid_labels: Optional[Sequence] = None,
        valid_transform: Optional[Callable] = _default_valid_transforms,
        test_filepaths: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        test_labels: Optional[Sequence] = None,
        loader: Callable = _pil_loader,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs
    ):
        """Creates a ImageClassificationData object from lists of image filepaths and labels

        Args:
            train_filepaths: sequence of file paths for training dataset. Defaults to None.
            train_labels: sequence of labels for training dataset. Defaults to None.
            train_transform: transforms for training dataset. Defaults to None.
            valid_filepaths: sequence of file paths for validation dataset. Defaults to None.
            valid_labels: sequence of labels for validation dataset. Defaults to None.
            valid_transform: transforms for validation and testing dataset. Defaults to None.
            test_filepaths: sequence of file paths for test dataset. Defaults to None.
            test_labels: sequence of labels for test dataset. Defaults to None.
            loader: function to load an image file. Defaults to None.
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

        Returns:
            ImageClassificationData: The constructed data module.

        Examples:
            >>> img_data = ImageClassificationData.from_filepaths(["a.png", "b.png"], [0, 1]) # doctest: +SKIP

        """
        dm =  super().from_filepaths(
                                    train_filepaths,
                                    train_labels,
                                    train_transform,
                                    valid_filepaths,
                                    valid_labels,
                                    valid_transform,
                                    test_filepaths,
                                    test_labels,
                                    loader,
                                    batch_size,
                                    num_workers,
                                    **kwargs)
        dm.data_pipeline = SpectrogramClassificationDataPipeline(
            train_transform=train_transform, valid_transform=valid_transform, loader=loader
        )
        return dm

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[Union[str, pathlib.Path]],
        train_transform: Optional[Callable] = _default_train_transforms,
        valid_folder: Optional[Union[str, pathlib.Path]] = None,
        valid_transform: Optional[Callable] = _default_valid_transforms,
        test_folder: Optional[Union[str, pathlib.Path]] = None,
        loader: Callable = _pil_loader,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **kwargs
    ):
        """
        Creates a ImageClassificationData object from folders of images arranged in this way: ::

            train/dog/xxx.png
            train/dog/xxy.png
            train/dog/xxz.png
            train/cat/123.png
            train/cat/nsdf3.png
            train/cat/asd932.png

        Args:
            train_folder: Path to training folder.
            train_transform: Image transform to use for training set.
            valid_folder: Path to validation folder.
            valid_transform: Image transform to use for validation and test set.
            test_folder: Path to test folder.
            loader: A function to load an image given its path.
            batch_size: Batch size for data loading.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

        Returns:
            ImageClassificationData: the constructed data module

        Examples:
            >>> img_data = ImageClassificationData.from_folders("train/") # doctest: +SKIP

        """
        dm = super().from_folders(
            train_folder,
            train_transform,
            valid_folder,
            valid_transform,
            test_folder,
            loader,
            batch_size,
            num_workers,
            **kwargs
        )
        dm.data_pipeline = SpectrogramClassificationDataPipeline(
            train_transform=train_transform, valid_transform=valid_transform, loader=loader
        )
        return dm

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, pathlib.Path],
        transform: Optional[Callable] = _default_valid_transforms,
        loader: Callable = _pil_loader,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs
    ):
        """
        Creates a ImageClassificationData object from folders of images arranged in this way: ::

            folder/dog_xxx.png
            folder/dog_xxy.png
            folder/dog_xxz.png
            folder/cat_123.png
            folder/cat_nsdf3.png
            folder/cat_asd932_.png

        Args:
            folder: Path to the data folder.
            transform: Image transform to apply to the data.
            loader: A function to load an image given its path.
            batch_size: Batch size for data loading.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

        Returns:
            ImageClassificationData: the constructed data module

        Examples:
            >>> img_data = ImageClassificationData.from_folder("my_folder/") # doctest: +SKIP

        """
        return super().from_folder(
            folder,
            transform,
            loader,
            batch_size,
            num_workers,
            **kwargs
        )
        dm.data_pipeline = SpectrogramClassificationDataPipeline(
            train_transform=train_transform, valid_transform=valid_transform, loader=loader
        )
        return dm
        
    @staticmethod
    def default_pipeline() -> SpectrogramClassificationDataPipeline:
        return SpectrogramClassificationDataPipeline(
            train_transform=_default_train_transforms, valid_transform=_default_valid_transforms, loader=_pil_loader
        )

  

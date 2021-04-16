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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import torch
import torchvision
from PIL import Image
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS, make_dataset

from flash.data.auto_dataset import AutoDataset
from flash.data.data_module import DataModule
from flash.data.data_pipeline import DataPipeline
from flash.data.process import Preprocess
from flash.utils.imports import _KORNIA_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as K
    import kornia.geometry.transform as T
else:
    from torchvision import transforms as T


class ImageClassificationPreprocess(Preprocess):

    to_tensor = torchvision.transforms.ToTensor()

    @staticmethod
    def _find_classes(dir: str) -> Tuple:
        """
        Finds the class folders in a dataset.
        Args:
            dir: Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def _get_predicting_files(samples: Union[Sequence, str]) -> List[str]:
        files = []
        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, (list, tuple)) and all(os.path.isdir(s) for s in samples):
            files = [os.path.join(sp, f) for sp in samples for f in os.listdir(sp)]

        elif isinstance(samples, (list, tuple)) and all(os.path.isfile(s) for s in samples):
            files = samples

        files = list(filter(lambda p: has_file_allowed_extension(p, IMG_EXTENSIONS), files))

        return files

    @classmethod
    def _load_data_dir(cls, data: Any, dataset: Optional[AutoDataset] = None) -> List[str]:
        if isinstance(data, list):
            dataset.num_classes = len(data)
            out = []
            for p, label in data:
                if os.path.isdir(p):
                    for f in os.listdir(p):
                        if has_file_allowed_extension(f, IMG_EXTENSIONS):
                            out.append([os.path.join(p, f), label])
                elif os.path.isfile(p) and has_file_allowed_extension(p, IMG_EXTENSIONS):
                    out.append([p, label])
            return out
        else:
            classes, class_to_idx = cls._find_classes(data)
            dataset.num_classes = len(classes)
            return make_dataset(data, class_to_idx, IMG_EXTENSIONS, None)

    @classmethod
    def _load_data_files_labels(cls, data: Any, dataset: Optional[AutoDataset] = None) -> Any:
        _classes = [tmp[1] for tmp in data]

        _classes = torch.stack([
            torch.tensor(int(_cls)) if not isinstance(_cls, torch.Tensor) else _cls.view(-1) for _cls in _classes
        ]).unique()

        dataset.num_classes = len(_classes)

        return data

    @classmethod
    def load_data(cls, data: Any, dataset: Optional[AutoDataset] = None) -> Iterable:
        if isinstance(data, (str, pathlib.Path, list)):
            return cls._load_data_dir(data=data, dataset=dataset)
        return cls._load_data_files_labels(data=data, dataset=dataset)

    @staticmethod
    def load_sample(sample) -> Union[Image.Image]:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        if isinstance(sample, torch.Tensor):
            return sample

        if isinstance(sample, (tuple, list)):
            path = sample[0]
            sample = list(sample)
        else:
            path = sample

        with open(path, "rb") as f, Image.open(f) as img:
            img = img.convert("RGB")

        if isinstance(sample, list):
            sample[0] = img
            return sample

        return img

    @classmethod
    def predict_load_data(cls, samples: Any) -> Iterable:
        if isinstance(samples, torch.Tensor):
            return samples
        return cls._get_predicting_files(samples)

    def collate(self, samples: Sequence) -> Any:
        _samples = []
        # todo: Kornia transforms add batch dimension which need to be removed
        for sample in samples:
            if isinstance(sample, tuple):
                sample = (sample[0].squeeze(0), ) + sample[1:]
            else:
                sample = sample.squeeze(0)
            _samples.append(sample)
        return default_collate(_samples)

    def common_step(self, sample: Any) -> Any:
        if isinstance(sample, (list, tuple)):
            source, target = sample
            return self.current_transform(source), target
        return self.current_transform(sample)

    def pre_tensor_transform(self, sample: Any) -> Any:
        return self.common_step(sample)

    def to_tensor_transform(self, sample: Any) -> Any:
        if self.current_transform == self._identify:
            if isinstance(sample, (list, tuple)):
                source, target = sample
                if isinstance(source, torch.Tensor):
                    return source, target
                return self.to_tensor(source), target
            elif isinstance(sample, torch.Tensor):
                return sample
            return self.to_tensor(sample)
        if isinstance(sample, torch.Tensor):
            return sample
        return self.common_step(sample)

    def post_tensor_transform(self, sample: Any) -> Any:
        return self.common_step(sample)

    def per_batch_transform(self, sample: Any) -> Any:
        return self.common_step(sample)

    def per_sample_transform_on_device(self, sample: Any) -> Any:
        return self.common_step(sample)

    def per_batch_transform_on_device(self, sample: Any) -> Any:
        return self.common_step(sample)


class ImageClassificationData(DataModule):
    """Data module for image classification tasks."""

    preprocess_cls = ImageClassificationPreprocess
    image_size = (196, 196)

    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        seed: int = 1234,
        train_split: Optional[Union[float, int]] = None,
        val_split: Optional[Union[float, int]] = None,
        test_split: Optional[Union[float, int]] = None,
        **kwargs,
    ) -> 'ImageClassificationData':
        """Creates a ImageClassificationData object from lists of image filepaths and labels"""

        if train_dataset is not None and train_split is not None or val_split is not None or test_split is not None:
            train_dataset, val_dataset, test_dataset = self.train_val_test_split(
                train_dataset, train_split, val_split, test_split, seed
            )

        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        self._num_classes = None

        if self._train_ds:
            self.set_dataset_attribute(self._train_ds, 'num_classes', self.num_classes)

        if self._val_ds:
            self.set_dataset_attribute(self._val_ds, 'num_classes', self.num_classes)

        if self._test_ds:
            self.set_dataset_attribute(self._test_ds, 'num_classes', self.num_classes)

        if self._predict_ds:
            self.set_dataset_attribute(self._predict_ds, 'num_classes', self.num_classes)

    @staticmethod
    def _check_transforms(transform: Dict[str, Union[nn.Module, Callable]]) -> Dict[str, Union[nn.Module, Callable]]:
        if transform and not isinstance(transform, Dict):
            raise MisconfigurationException(
                "Transform should be a dict. "
                f"Here are the available keys for your transforms: {DataPipeline.PREPROCESS_FUNCS}."
            )
        if "per_batch_transform" in transform and "per_sample_transform_on_device" in transform:
            raise MisconfigurationException(
                f'{transform}: `per_batch_transform` and `per_sample_transform_on_device` '
                f'are mutually exclusive.'
            )
        return transform

    @staticmethod
    def default_train_transforms():
        image_size = ImageClassificationData.image_size
        if _KORNIA_AVAILABLE and not os.getenv("FLASH_TESTING", "0") == "1":
            #  Better approach as all transforms are applied on tensor directly
            return {
                "to_tensor_transform": torchvision.transforms.ToTensor(),
                "post_tensor_transform": nn.Sequential(K.RandomResizedCrop(image_size), K.RandomHorizontalFlip()),
                "per_batch_transform_on_device": nn.Sequential(
                    K.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
                )
            }
        else:
            from torchvision import transforms as T  # noqa F811
            return {
                "pre_tensor_transform": nn.Sequential(T.RandomResizedCrop(image_size), T.RandomHorizontalFlip()),
                "to_tensor_transform": torchvision.transforms.ToTensor(),
                "post_tensor_transform": T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            }

    @staticmethod
    def default_val_transforms():
        image_size = ImageClassificationData.image_size
        if _KORNIA_AVAILABLE and not os.getenv("FLASH_TESTING", "0") == "1":
            #  Better approach as all transforms are applied on tensor directly
            return {
                "to_tensor_transform": torchvision.transforms.ToTensor(),
                "post_tensor_transform": nn.Sequential(K.RandomResizedCrop(image_size)),
                "per_batch_transform_on_device": nn.Sequential(
                    K.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
                )
            }
        else:
            from torchvision import transforms as T  # noqa F811
            return {
                "pre_tensor_transform": T.Compose([T.RandomResizedCrop(image_size)]),
                "to_tensor_transform": torchvision.transforms.ToTensor(),
                "post_tensor_transform": T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            }

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
            if self._train_ds is not None:
                self._num_classes = self._get_num_classes(self._train_ds)

        return self._num_classes

    def _get_num_classes(self, dataset: torch.utils.data.Dataset):
        num_classes = self.get_dataset_attribute(dataset, "num_classes", None)
        if num_classes is None:
            num_classes = torch.tensor([dataset[idx][1] for idx in range(len(dataset))]).unique().numel()

        return num_classes

    @classmethod
    def instantiate_preprocess(
        cls,
        train_transform: Dict[str, Union[nn.Module, Callable]],
        val_transform: Dict[str, Union[nn.Module, Callable]],
        test_transform: Dict[str, Union[nn.Module, Callable]],
        predict_transform: Dict[str, Union[nn.Module, Callable]],
        preprocess_cls: Type[Preprocess] = None
    ) -> Preprocess:
        """
        This function is used to instantiate ImageClassificationData preprocess object.

        Args:
            train_transform: Train transforms for images.
            val_transform: Validation transforms for images.
            test_transform: Test transforms for images.
            predict_transform: Predict transforms for images.
            preprocess_cls: User provided preprocess_cls.

        Example::

            train_transform = {
                "per_sample_transform": T.Compose([
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
                "per_batch_transform_on_device": nn.Sequential(K.RandomAffine(360), K.ColorJitter(0.2, 0.3, 0.2, 0.3))
            }

        """
        train_transform, val_transform, test_transform, predict_transform = cls._resolve_transforms(
            train_transform, val_transform, test_transform, predict_transform
        )

        preprocess_cls = preprocess_cls or cls.preprocess_cls
        preprocess: Preprocess = preprocess_cls(train_transform, val_transform, test_transform, predict_transform)
        return preprocess

    @classmethod
    def _resolve_transforms(
        cls,
        train_transform: Optional[Union[str, Dict]] = 'default',
        val_transform: Optional[Union[str, Dict]] = 'default',
        test_transform: Optional[Union[str, Dict]] = 'default',
        predict_transform: Optional[Union[str, Dict]] = 'default',
    ):

        if not train_transform or train_transform == 'default':
            train_transform = cls.default_train_transforms()

        if not val_transform or val_transform == 'default':
            val_transform = cls.default_val_transforms()

        if not test_transform or test_transform == 'default':
            test_transform = cls.default_val_transforms()

        if not predict_transform or predict_transform == 'default':
            predict_transform = cls.default_val_transforms()

        return (
            cls._check_transforms(train_transform), cls._check_transforms(val_transform),
            cls._check_transforms(test_transform), cls._check_transforms(predict_transform)
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[Union[str, pathlib.Path]] = None,
        val_folder: Optional[Union[str, pathlib.Path]] = None,
        test_folder: Optional[Union[str, pathlib.Path]] = None,
        predict_folder: Union[str, pathlib.Path] = None,
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
        Creates a ImageClassificationData object from folders of images arranged in this way: ::

            train/dog/xxx.png
            train/dog/xxy.png
            train/dog/xxz.png
            train/cat/123.png
            train/cat/nsdf3.png
            train/cat/asd932.png

        Args:
            train_folder: Path to training folder. Default: None.
            val_folder: Path to validation folder. Default: None.
            test_folder: Path to test folder. Default: None.
            predict_folder: Path to predict folder. Default: None.
            val_transform: Image transform to use for validation and test set.
            train_transform: Image transform to use for training set.
            val_transform: Image transform to use for validation set.
            test_transform: Image transform to use for test set.
            predict_transform: Image transform to use for predict set.
            batch_size: Batch size for data loading.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to ``None`` which equals the number of available CPU threads.

        Returns:
            ImageClassificationData: the constructed data module

        Examples:
            >>> img_data = ImageClassificationData.from_folders("train/") # doctest: +SKIP

        """
        preprocess = cls.instantiate_preprocess(
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

    @classmethod
    def from_filepaths(
        cls,
        train_filepaths: Optional[Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]]] = None,
        train_labels: Optional[Sequence] = None,
        val_filepaths: Optional[Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]]] = None,
        val_labels: Optional[Sequence] = None,
        test_filepaths: Optional[Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]]] = None,
        test_labels: Optional[Sequence] = None,
        predict_filepaths: Optional[Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]]] = None,
        train_transform: Union[str, Dict] = 'default',
        val_transform: Union[str, Dict] = 'default',
        test_transform: Union[str, Dict] = 'default',
        predict_transform: Union[str, Dict] = 'default',
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        seed: Optional[int] = 42,
        preprocess_cls: Optional[Type[Preprocess]] = None,
        **kwargs,
    ) -> 'ImageClassificationData':
        """
        Creates a ImageClassificationData object from folders of images arranged in this way: ::

            folder/dog_xxx.png
            folder/dog_xxy.png
            folder/dog_xxz.png
            folder/cat_123.png
            folder/cat_nsdf3.png
            folder/cat_asd932_.png

        Args:

            train_filepaths: String or sequence of file paths for training dataset. Defaults to ``None``.
            train_labels: Sequence of labels for training dataset. Defaults to ``None``.
            val_filepaths: String or sequence of file paths for validation dataset. Defaults to ``None``.
            val_labels: Sequence of labels for validation dataset. Defaults to ``None``.
            test_filepaths: String or sequence of file paths for test dataset. Defaults to ``None``.
            test_labels: Sequence of labels for test dataset. Defaults to ``None``.
            train_transform: Transforms for training dataset. Defaults to ``default``,
                which loads imagenet transforms.
            val_transform: Transforms for validation and testing dataset.
                Defaults to ``default``, which loads imagenet transforms.
            batch_size: The batchsize to use for parallel loading. Defaults to ``64``.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to ``None`` which equals the number of available CPU threads.
            seed: Used for the train/val splits.

        Returns:

            ImageClassificationData: The constructed data module.
        """
        # enable passing in a string which loads all files in that folder as a list
        if isinstance(train_filepaths, str):
            if os.path.isdir(train_filepaths):
                train_filepaths = [os.path.join(train_filepaths, x) for x in os.listdir(train_filepaths)]
            else:
                train_filepaths = [train_filepaths]

        if isinstance(val_filepaths, str):
            if os.path.isdir(val_filepaths):
                val_filepaths = [os.path.join(val_filepaths, x) for x in os.listdir(val_filepaths)]
            else:
                val_filepaths = [val_filepaths]

        if isinstance(test_filepaths, str):
            if os.path.isdir(test_filepaths):
                test_filepaths = [os.path.join(test_filepaths, x) for x in os.listdir(test_filepaths)]
            else:
                test_filepaths = [test_filepaths]

        preprocess = cls.instantiate_preprocess(
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
            preprocess_cls=preprocess_cls,
        )

        return cls.from_load_data_inputs(
            train_load_data_input=list(zip(train_filepaths, train_labels)) if train_filepaths else None,
            val_load_data_input=list(zip(val_filepaths, val_labels)) if val_filepaths else None,
            test_load_data_input=list(zip(test_filepaths, test_labels)) if test_filepaths else None,
            predict_load_data_input=predict_filepaths,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            seed=seed,
            **kwargs
        )

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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from numpy import isin
from PIL import Image
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.nn.modules import ModuleDict
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms as torchvision_T
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS, make_dataset
from torchvision.transforms.functional import to_pil_image

from flash.core.imports import _KORNIA_AVAILABLE
from flash.data.auto_dataset import AutoDataset
from flash.data.data_module import DataModule
from flash.data.data_pipeline import DataPipeline
from flash.data.process import Preprocess
from flash.data.utils import _contains_any_tensor

if _KORNIA_AVAILABLE:
    import kornia.augmentation as K
    import kornia.geometry.transform as T
else:
    from torchvision import transforms as T


class ImageClassificationPreprocess(Preprocess):
    to_tensor = torchvision_T.ToTensor()

    @staticmethod
    def _find_classes(dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

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
    def _get_predicting_files(samples):
        files = []
        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, list) and all(os.path.isdir(s) for s in samples):
            for s in samples:
                for f in os.listdir(s):
                    files += [os.path.join(s, f)]

        elif isinstance(samples, list) and all(os.path.isfile(s) for s in samples):
            files = samples

        files = list(filter(lambda p: has_file_allowed_extension(p, IMG_EXTENSIONS), files))

        return files

    @classmethod
    def _load_data_dir(cls, data: Any, dataset: Optional[AutoDataset] = None):
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
            print(out)
            return out
        else:
            classes, class_to_idx = cls._find_classes(data)
            dataset.num_classes = len(classes)
            return make_dataset(data, class_to_idx, IMG_EXTENSIONS, None)

    @classmethod
    def _load_data_files_labels(cls, data: Any, dataset: Optional[AutoDataset] = None):
        _classes = [tmp[1] for tmp in data]

        _classes = torch.stack([
            torch.tensor(int(_cls)) if not isinstance(_cls, torch.Tensor) else _cls.view(-1) for _cls in _classes
        ]).unique()

        dataset.num_classes = len(_classes)

        return data

    @classmethod
    def load_data(cls, data: Any, dataset: Optional[AutoDataset] = None) -> Any:
        if isinstance(data, (str, pathlib.Path, list)):
            return cls._load_data_dir(data=data, dataset=dataset)
        return cls._load_data_files_labels(data=data, dataset=dataset)

    @staticmethod
    def load_sample(sample) -> Union[Image.Image, list]:
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
    def predict_load_data(cls, samples: Any) -> Any:
        if isinstance(samples, torch.Tensor):
            return samples
        return cls._get_predicting_files(samples)

    def _convert_tensor_to_pil(self, sample):
        #  some datasets provide their data as tensors.
        # however, it would be better to convert those data once in load_data
        if isinstance(sample, torch.Tensor):
            sample = to_pil_image(sample)
        return sample

    def _apply_transform(
        self, sample: Any, transform: Union[Callable, Dict[str, Callable]], func_name: str
    ) -> torch.Tensor:
        if transform is not None:
            if isinstance(transform, (Dict, ModuleDict)):
                if func_name not in transform:
                    return sample
                else:
                    transform = transform[func_name]
            sample = transform(sample)
        return sample

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

    def common_per_sample_pre_tensor_transform(self, sample: Any, transform) -> Any:
        return self._apply_transform(sample, transform, "per_sample_pre_tensor_transform")

    def train_per_sample_pre_tensor_transform(self, sample: Any) -> Any:
        sample, target = sample
        return self.common_per_sample_pre_tensor_transform(sample, self.train_transform), target

    def val_per_sample_pre_tensor_transform(self, sample: Any) -> Any:
        sample, target = sample
        return self.common_per_sample_pre_tensor_transform(sample, self.valid_transform), target

    def test_per_sample_pre_tensor_transform(self, sample: Any) -> Any:
        sample, target = sample
        return self.common_per_sample_pre_tensor_transform(sample, self.test_transform), target

    def predict_per_sample_pre_tensor_transform(self, sample: Any) -> Any:
        if isinstance(sample, torch.Tensor):
            return sample
        return self.common_per_sample_pre_tensor_transform(sample, self.predict_transform)

    def per_sample_to_tensor_transform(self, sample) -> Any:
        sample, target = sample
        return sample if isinstance(sample, torch.Tensor) else self.to_tensor(sample), target

    def predict_per_sample_to_tensor_transform(self, sample) -> Any:
        if isinstance(sample, torch.Tensor):
            return sample
        return self.to_tensor(sample)

    def common_per_sample_post_tensor_transform(self, sample: Any, transform) -> Any:
        return self._apply_transform(sample, transform, "per_sample_post_tensor_transform")

    def train_per_sample_post_tensor_transform(self, sample: Any) -> Any:
        sample, target = sample
        return self.common_per_sample_post_tensor_transform(sample, self.train_transform), target

    def val_per_sample_post_tensor_transform(self, sample: Any) -> Any:
        sample, target = sample
        return self.common_per_sample_post_tensor_transform(sample, self.valid_transform), target

    def test_per_sample_post_tensor_transform(self, sample: Any) -> Any:
        sample, target = sample
        return self.common_per_sample_post_tensor_transform(sample, self.test_transform), target

    def predict_per_sample_post_tensor_transform(self, sample: Any) -> Any:
        return self.common_per_sample_post_tensor_transform(sample, self.predict_transform)

    def train_per_batch_transform_on_device(self, batch: Tuple) -> Tuple:
        batch, target = batch
        return self._apply_transform(batch, self.train_transform, "per_batch_transform_on_device"), target


class ImageClassificationData(DataModule):
    """Data module for image classification tasks."""

    preprocess_cls = ImageClassificationPreprocess
    image_size = (196, 196)

    def __init__(
        self,
        train_ds: Optional[torch.utils.data.Dataset] = None,
        valid_ds: Optional[torch.utils.data.Dataset] = None,
        test_ds: Optional[torch.utils.data.Dataset] = None,
        predict_ds: Optional[torch.utils.data.Dataset] = None,
        train_transform: Optional[Union[str, Dict]] = 'default',
        valid_transform: Optional[Union[str, Dict]] = 'default',
        test_transform: Optional[Union[str, Dict]] = 'default',
        predict_transform: Optional[Union[str, Dict]] = 'default',
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        train_split: Optional[Union[float, int]] = None,
        valid_split: Optional[Union[float, int]] = None,
        test_split: Optional[Union[float, int]] = None,
        seed: Optional[int] = 1234,
    ):

        if train_ds is not None and train_split is not None or valid_split is not None or test_split is not None:
            train_ds, _valid_ds, _test_ds = self.train_valid_test_split(
                train_ds, train_split, valid_split, test_split, seed
            )

            if _valid_ds is not None:
                valid_ds = _valid_ds

            if _test_ds is not None:
                test_ds = _test_ds

        super().__init__(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            predict_ds=predict_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self._num_classes = None

        if self._train_ds is not None:
            self.set_dataset_attribute(self._train_ds, 'num_classes', self.num_classes)

        if self._valid_ds is not None:
            self.set_dataset_attribute(self._valid_ds, 'num_classes', self.num_classes)

        if self._test_ds is not None:
            self.set_dataset_attribute(self._test_ds, 'num_classes', self.num_classes)

        if self._predict_ds is not None:
            self.set_dataset_attribute(self._predict_ds, 'num_classes', self.num_classes)

        if isinstance(train_transform, str) and train_transform == 'default':
            train_transform = self.default_train_transforms()

        if isinstance(valid_transform, str) and valid_transform == 'default':
            valid_transform = self.default_valid_transforms()

        if isinstance(test_transform, str) and test_transform == 'default':
            test_transform = self.default_valid_transforms()

        if isinstance(predict_transform, str) and predict_transform == 'default':
            predict_transform = self.default_valid_transforms()

        self.train_transform = self._check_transforms(train_transform)
        self.valid_transform = self._check_transforms(valid_transform)
        self.test_transform = self._check_transforms(test_transform)
        self.predict_transform = self._check_transforms(predict_transform)

    @staticmethod
    def _check_transforms(transform: dict) -> dict:
        if transform is not None and not isinstance(transform, dict):
            raise MisconfigurationException(
                "Transform should be a dict. "
                f"Here are the available keys for your transforms: {DataPipeline.PREPROCESS_FUNCS}."
            )
        return transform

    @staticmethod
    def default_train_transforms():
        image_size = ImageClassificationData.image_size
        if _KORNIA_AVAILABLE:
            #  Better approach as all transforms are applied on tensor directly
            return {
                "per_sample_post_tensor_transform": nn.Sequential(
                    K.RandomResizedCrop(image_size), K.RandomHorizontalFlip()
                ),
                "per_batch_transform_on_device": nn.Sequential(
                    K.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
                )
            }
        else:
            return {
                "per_sample_pre_tensor_transform": nn.Sequential(
                    T.RandomResizedCrop(image_size), T.RandomHorizontalFlip()
                ),
                "per_sample_post_tensor_transform": T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            }

    @staticmethod
    def default_valid_transforms():
        image_size = ImageClassificationData.image_size
        if _KORNIA_AVAILABLE:
            #  Better approach as all transforms are applied on tensor directly
            return {
                "per_sample_post_tensor_transform": nn.Sequential(K.RandomResizedCrop(image_size)),
                "per_batch_transform_on_device": nn.Sequential(
                    K.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
                )
            }
        else:
            return {
                "per_sample_pre_tensor_transform": T.Compose([T.RandomResizedCrop(image_size)]),
                "per_sample_post_tensor_transform": T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            }

    @property
    def num_classes(self):
        if self._num_classes is None:
            if self._train_ds is not None:
                self._num_classes = self._get_num_classes(self._train_ds)

        return self._num_classes

    def _get_num_classes(self, dataset: torch.utils.data.Dataset):
        num_classes = self.get_dataset_attribute(dataset, "num_classes", None)
        if num_classes is None:
            num_classes = torch.tensor([dataset[idx][1] for idx in range(len(dataset))]).unique().numel()

        return num_classes

    @property
    def preprocess(self) -> ImageClassificationPreprocess:
        return self.preprocess_cls(
            train_transform=self.train_transform,
            valid_transform=self.valid_transform,
            test_transform=self.test_transform,
            predict_transform=self.predict_transform
        )

    @classmethod
    def _generate_dataset_if_possible(
        cls,
        data: Optional[Any],
        running_stage: RunningStage,
        whole_data_load_fn: Optional[Callable] = None,
        per_sample_load_fn: Optional[Callable] = None,
        data_pipeline: Optional[DataPipeline] = None
    ) -> Optional[AutoDataset]:
        if data is None:
            return None

        if data_pipeline is not None:
            return data_pipeline._generate_auto_dataset(data, running_stage=running_stage)

        return cls.autogenerate_dataset(data, running_stage, whole_data_load_fn, per_sample_load_fn, data_pipeline)

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[Union[str, pathlib.Path]] = None,
        valid_folder: Optional[Union[str, pathlib.Path]] = None,
        test_folder: Optional[Union[str, pathlib.Path]] = None,
        predict_folder: Union[str, pathlib.Path] = None,
        train_transform: Optional[Union[str, Dict]] = 'default',
        valid_transform: Optional[Union[str, Dict]] = 'default',
        test_transform: Optional[Union[str, Dict]] = 'default',
        predict_transform: Optional[Union[str, Dict]] = 'default',
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        data_pipeline: Optional[DataPipeline] = None,
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
            valid_folder: Path to validation folder.
            test_folder: Path to test folder.
            predict: Path to predict folder.
            valid_transform: Image transform to use for validation and test set.
            train_transform: Image transform to use for training set.
            batch_size: Batch size for data loading.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to ``None`` which equals the number of available CPU threads.

        Returns:
            ImageClassificationData: the constructed data module

        Examples:
            >>> img_data = ImageClassificationData.from_folders("train/") # doctest: +SKIP



        """
        return cls.from_load_data_inputs(
            train_load_data_input=train_folder,
            valid_load_data_input=valid_folder,
            test_load_data_input=test_folder,
            predict_load_data_input=predict_folder,
            train_transform=train_transform,
            valid_transform=valid_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @classmethod
    def from_filepaths(
        cls,
        train_filepaths: Union[str, Optional[Sequence[Union[str, pathlib.Path]]]] = None,
        train_labels: Optional[Sequence] = None,
        valid_filepaths: Union[str, Optional[Sequence[Union[str, pathlib.Path]]]] = None,
        valid_labels: Optional[Sequence] = None,
        test_filepaths: Union[str, Optional[Sequence[Union[str, pathlib.Path]]]] = None,
        test_labels: Optional[Sequence] = None,
        predict_filepaths: Union[str, Optional[Sequence[Union[str, pathlib.Path]]]] = None,
        train_transform: Optional[Callable] = 'default',
        valid_transform: Optional[Callable] = 'default',
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        seed: int = 1234,
        **kwargs
    ):
        """Creates a ImageClassificationData object from lists of image filepaths and labels

        Args:
            train_filepaths: string or sequence of file paths for training dataset. Defaults to ``None``.
            train_labels: sequence of labels for training dataset. Defaults to ``None``.
            valid_split: if not None, generates val split from train dataloader using this value.
            valid_filepaths: string or sequence of file paths for validation dataset. Defaults to ``None``.
            valid_labels: sequence of labels for validation dataset. Defaults to ``None``.
            test_filepaths: string or sequence of file paths for test dataset. Defaults to ``None``.
            test_labels: sequence of labels for test dataset. Defaults to ``None``.
            train_transform: transforms for training dataset. Defaults to ``default``, which loads imagenet transforms.
            valid_transform: transforms for validation and testing dataset.
                Defaults to ``default``, which loads imagenet transforms.
            batch_size: the batchsize to use for parallel loading. Defaults to ``64``.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to ``None`` which equals the number of available CPU threads.
            seed: Used for the train/val splits when valid_split is not None

        Returns:
            ImageClassificationData: The constructed data module.

        Examples:
            >>> img_data = ImageClassificationData.from_filepaths(["a.png", "b.png"], [0, 1]) # doctest: +SKIP

        Example when labels are in .csv file::

            train_labels = labels_from_categorical_csv('path/to/train.csv', 'my_id')
            valid_labels = labels_from_categorical_csv(path/to/valid.csv', 'my_id')
            test_labels = labels_from_categorical_csv(path/to/tests.csv', 'my_id')

            data = ImageClassificationData.from_filepaths(
                batch_size=2,
                train_filepaths='path/to/train',
                train_labels=train_labels,
                valid_filepaths='path/to/valid',
                valid_labels=valid_labels,
                test_filepaths='path/to/test',
                test_labels=test_labels,
            )

        """
        # enable passing in a string which loads all files in that folder as a list
        if isinstance(train_filepaths, str):
            if os.path.isdir(train_filepaths):
                train_filepaths = [os.path.join(train_filepaths, x) for x in os.listdir(train_filepaths)]
            else:
                train_filepaths = [train_filepaths]
        if isinstance(valid_filepaths, str):
            if os.path.isdir(valid_filepaths):
                valid_filepaths = [os.path.join(valid_filepaths, x) for x in os.listdir(valid_filepaths)]
            else:
                valid_filepaths = [valid_filepaths]
        if isinstance(test_filepaths, str):
            if os.path.isdir(test_filepaths):
                test_filepaths = [os.path.join(test_filepaths, x) for x in os.listdir(test_filepaths)]
            else:
                test_filepaths = [test_filepaths]
        if isinstance(predict_filepaths, str):
            if os.path.isdir(predict_filepaths):
                predict_filepaths = [os.path.join(predict_filepaths, x) for x in os.listdir(predict_filepaths)]
            else:
                predict_filepaths = [predict_filepaths]

        if train_filepaths is not None and train_labels is not None:
            train_ds = cls._generate_dataset_if_possible(
                list(zip(train_filepaths, train_labels)), running_stage=RunningStage.TRAINING
            )
        else:
            train_ds = None

        if valid_filepaths is not None and valid_labels is not None:
            valid_ds = cls._generate_dataset_if_possible(
                list(zip(valid_filepaths, valid_labels)), running_stage=RunningStage.VALIDATING
            )
        else:
            valid_ds = None

        if test_filepaths is not None and test_labels is not None:
            test_ds = cls._generate_dataset_if_possible(
                list(zip(test_filepaths, test_labels)), running_stage=RunningStage.TESTING
            )
        else:
            test_ds = None

        if predict_filepaths is not None:
            predict_ds = cls._generate_dataset_if_possible(predict_filepaths, running_stage=RunningStage.PREDICTING)
        else:
            predict_ds = None

        return cls(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            predict_ds=predict_ds,
            train_transform=train_transform,
            valid_transform=valid_transform,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            **kwargs
        )

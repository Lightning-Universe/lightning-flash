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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms as T
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS, make_dataset

from flash.core.classification import ClassificationState
from flash.core.utils import _is_overriden
from flash.data.auto_dataset import AutoDataset
from flash.data.base_viz import BaseVisualization  # for viz
from flash.data.callback import BaseDataFetcher
from flash.data.data_module import DataModule
from flash.data.process import Preprocess
from flash.data.utils import _PREPROCESS_FUNCS
from flash.utils.imports import _KORNIA_AVAILABLE, _MATPLOTLIB_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia as K

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None


class ImageClassificationPreprocess(Preprocess):

    to_tensor = T.ToTensor()

    def __init__(
        self,
        train_transform: Optional[Union[Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Dict[str, Callable]]] = None,
        predict_transform: Optional[Union[Dict[str, Callable]]] = None,
        image_size: Tuple[int, int] = (196, 196),
    ):
        """
        Preprocess pipeline definition for image classification tasks.

        Args:
            train_transform: Dictionary with the set of transform to apply during training. Default: None.
            val_transform: Dictionary with the set of transform to apply during validation. Default: None.
            test_transform: Dictionary with the set of transform to apply during testing. Default: None.
            predict_transform: Dictionary with the set of transform to apply during prediction. Default: None.
            image_size: A tuple with the expected output image size. Default: (196, 196).

        Returns:
            ImageClassificationPreprocess: The constructed preprocess module.
        """
        train_transform, val_transform, test_transform, predict_transform = self._resolve_transforms(
            train_transform, val_transform, test_transform, predict_transform, image_size
        )
        super().__init__(train_transform, val_transform, test_transform, predict_transform)

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

    def default_train_transforms(self, image_size: Tuple[int, int]) -> Dict[str, Callable]:
        if _KORNIA_AVAILABLE and not os.getenv("FLASH_TESTING", "0") == "1":
            #  Better approach as all transforms are applied on tensor directly
            return {
                "to_tensor_transform": torchvision.transforms.ToTensor(),
                "post_tensor_transform": nn.Sequential(
                    K.augmentation.RandomResizedCrop(image_size), K.augmentation.RandomHorizontalFlip()
                ),
                "per_batch_transform_on_device": nn.Sequential(
                    K.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
                )
            }
        else:
            from torchvision import transforms as T  # noqa F811
            return {
                "pre_tensor_transform": nn.Sequential(T.RandomResizedCrop(image_size), T.RandomHorizontalFlip()),
                "to_tensor_transform": torchvision.transforms.ToTensor(),
                "post_tensor_transform": T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            }

    def default_val_transforms(self, image_size: Tuple[int, int]) -> Dict[str, Callable]:
        if _KORNIA_AVAILABLE and not os.getenv("FLASH_TESTING", "0") == "1":
            #  Better approach as all transforms are applied on tensor directly
            return {
                "to_tensor_transform": torchvision.transforms.ToTensor(),
                "post_tensor_transform": nn.Sequential(K.augmentation.RandomResizedCrop(image_size)),
                "per_batch_transform_on_device": nn.Sequential(
                    K.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
                )
            }
        else:
            from torchvision import transforms as T  # noqa F811
            return {
                "pre_tensor_transform": T.Compose([T.RandomResizedCrop(image_size)]),
                "to_tensor_transform": torchvision.transforms.ToTensor(),
                "post_tensor_transform": T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            }

    def _resolve_transforms(
        self,
        train_transform: Optional[Union[str, Dict]] = 'default',
        val_transform: Optional[Union[str, Dict]] = 'default',
        test_transform: Optional[Union[str, Dict]] = 'default',
        predict_transform: Optional[Union[str, Dict]] = 'default',
        image_size: Tuple[int, int] = (196, 196),
    ):

        if not train_transform or train_transform == 'default':
            train_transform = self.default_train_transforms(image_size)

        if not val_transform or val_transform == 'default':
            val_transform = self.default_val_transforms(image_size)

        if not test_transform or test_transform == 'default':
            test_transform = self.default_val_transforms(image_size)

        if not predict_transform or predict_transform == 'default':
            predict_transform = self.default_val_transforms(image_size)

        return (
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
        )

    @classmethod
    def _load_data_dir(
        cls,
        data: Any,
        dataset: Optional[AutoDataset] = None,
    ) -> Tuple[Optional[List[str]], List[Tuple[str, int]]]:
        if isinstance(data, list):
            # TODO: define num_classes elsewhere. This is a bad assumption since the list of
            # labels might not contain the complete set of ids so that you can infer the total
            # number of classes to train in your dataset.
            dataset.num_classes = len(data)
            out: List[Tuple[str, int]] = []
            for p, label in data:
                if os.path.isdir(p):
                    # TODO: there is an issue here when a path is provided along with labels.
                    # os.listdir cannot assure the same file order as the passed labels list.
                    files_list: List[str] = os.listdir(p)
                    if len(files_list) > 1:
                        raise ValueError(
                            f"The provided directory contains more than one file."
                            f"Directory: {p} -> Contains: {files_list}"
                        )
                    for f in files_list:
                        if has_file_allowed_extension(f, IMG_EXTENSIONS):
                            out.append([os.path.join(p, f), label])
                elif os.path.isfile(p) and has_file_allowed_extension(str(p), IMG_EXTENSIONS):
                    out.append([p, label])
                else:
                    raise TypeError(f"Unexpected file path type: {p}.")
            return None, out
        else:
            classes, class_to_idx = cls._find_classes(data)
            # TODO: define num_classes elsewhere. This is a bad assumption since the list of
            # labels might not contain the complete set of ids so that you can infer the total
            # number of classes to train in your dataset.
            dataset.num_classes = len(classes)
            return classes, make_dataset(data, class_to_idx, IMG_EXTENSIONS, None)

    @classmethod
    def _load_data_files_labels(cls, data: Any, dataset: Optional[AutoDataset] = None) -> Any:
        _classes = [tmp[1] for tmp in data]

        _classes = torch.stack([
            torch.tensor(int(_cls)) if not isinstance(_cls, torch.Tensor) else _cls.view(-1) for _cls in _classes
        ]).unique()

        dataset.num_classes = len(_classes)

        return data

    def load_data(self, data: Any, dataset: Optional[AutoDataset] = None) -> Iterable:
        if isinstance(data, (str, pathlib.Path, list)):
            classes, data = self._load_data_dir(data=data, dataset=dataset)
            state = ClassificationState(classes)
            self.set_state(state)
            return data
        return self._load_data_files_labels(data=data, dataset=dataset)

    @staticmethod
    def load_sample(sample) -> Union[Image.Image, torch.Tensor, Tuple[Image.Image, torch.Tensor]]:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        if isinstance(sample, torch.Tensor):
            out: torch.Tensor = sample
            return out

        path: str = ""
        if isinstance(sample, (tuple, list)):
            path = sample[0]
            sample = list(sample)
        else:
            path = sample

        with open(path, "rb") as f, Image.open(f) as img:
            img_out: Image.Image = img.convert("RGB")

        if isinstance(sample, list):
            # return a tuple with the PIL image and tensor with the labels.
            # returning the tensor helps later to easily collate the batch
            # for single/multi label at the same time.
            out: Tuple[Image.Image, torch.Tensor] = (img_out, torch.as_tensor(sample[1]))
            return out

        return img_out

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
        if self.current_transform == self._identity:
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

    def per_batch_transform_on_device(self, sample: Any) -> Any:
        return self.common_step(sample)


class ImageClassificationData(DataModule):
    """Data module for image classification tasks."""

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return MatplotlibVisualization(*args, **kwargs)

    def _get_num_classes(self, dataset: torch.utils.data.Dataset):
        num_classes = self.get_dataset_attribute(dataset, "num_classes", None)
        if num_classes is None:
            num_classes = torch.tensor([dataset[idx][1] for idx in range(len(dataset))]).unique().numel()

        return num_classes

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
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
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
        preprocess = preprocess or ImageClassificationPreprocess(
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
        )

        return cls.from_load_data_inputs(
            train_load_data_input=train_folder,
            val_load_data_input=val_folder,
            test_load_data_input=test_folder,
            predict_load_data_input=predict_folder,
            batch_size=batch_size,
            num_workers=num_workers,
            data_fetcher=data_fetcher,
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
        image_size: Tuple[int, int] = (196, 196),
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        seed: Optional[int] = 42,
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
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
            train_transform: Image transform to use for the train set. Defaults to ``default``, which loads imagenet
                transforms.
            val_transform: Image transform to use for the validation set. Defaults to ``default``, which loads
                imagenet transforms.
            test_transform: Image transform to use for the test set. Defaults to ``default``, which loads imagenet
                transforms.
            predict_transform: Image transform to use for the predict set. Defaults to ``default``, which loads imagenet
                transforms.
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

        preprocess = preprocess or ImageClassificationPreprocess(
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
            image_size=image_size,
        )

        return cls.from_load_data_inputs(
            train_load_data_input=list(zip(train_filepaths, train_labels)) if train_filepaths else None,
            val_load_data_input=list(zip(val_filepaths, val_labels)) if val_filepaths else None,
            test_load_data_input=list(zip(test_filepaths, test_labels)) if test_filepaths else None,
            predict_load_data_input=predict_filepaths,
            batch_size=batch_size,
            num_workers=num_workers,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            seed=seed,
            val_split=val_split,
            **kwargs
        )


class MatplotlibVisualization(BaseVisualization):
    """Process and show the image batch and its associated label using matplotlib.
    """
    max_cols: int = 4  # maximum number of columns we accept
    block_viz_window: bool = True  # parameter to allow user to block visualisation windows

    @staticmethod
    def _to_numpy(img: Union[torch.Tensor, Image.Image]) -> np.ndarray:
        out: np.ndarray
        if isinstance(img, Image.Image):
            out = np.array(img)
        elif isinstance(img, torch.Tensor):
            out = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError(f"Unknown image type. Got: {type(img)}.")
        return out

    def _show_images_and_labels(self, data: List[Any], num_samples: int, title: str):
        # define the image grid
        cols: int = min(num_samples, self.max_cols)
        rows: int = num_samples // cols

        if not _MATPLOTLIB_AVAILABLE:
            raise MisconfigurationException("You need matplotlib to visualise. Please, pip install matplotlib")

        # create figure and set title
        fig, axs = plt.subplots(rows, cols)
        fig.suptitle(title)

        for i, ax in enumerate(axs.ravel()):
            # unpack images and labels
            if isinstance(data, list):
                _img, _label = data[i]
            elif isinstance(data, tuple):
                imgs, labels = data
                _img, _label = imgs[i], labels[i]
            else:
                raise TypeError(f"Unknown data type. Got: {type(data)}.")
            # convert images to numpy
            _img: np.ndarray = self._to_numpy(_img)
            if isinstance(_label, torch.Tensor):
                _label = _label.squeeze().tolist()
            # show image and set label as subplot title
            ax.imshow(_img)
            ax.set_title(str(_label))
            ax.axis('off')
        plt.show(block=self.block_viz_window)

    def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_load_sample"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_pre_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_pre_tensor_transform"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_to_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_to_tensor_transform"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_post_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_post_tensor_transform"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_per_batch_transform(self, batch: List[Any], running_stage):
        win_title: str = f"{running_stage} - show_per_batch_transform"
        self._show_images_and_labels(batch[0], batch[0][0].shape[0], win_title)

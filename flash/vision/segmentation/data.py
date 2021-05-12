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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS

import flash
from flash.data.auto_dataset import BaseAutoDataset
from flash.data.base_viz import BaseVisualization  # for viz
from flash.data.callback import BaseDataFetcher
from flash.data.data_module import DataModule
from flash.data.data_source import (
    DefaultDataKeys,
    DefaultDataSources,
    ImageLabelsMap,
    NumpyDataSource,
    PathsDataSource,
    TensorDataSource,
)
from flash.data.process import Preprocess
from flash.utils.imports import _MATPLOTLIB_AVAILABLE
from flash.vision.segmentation.serialization import SegmentationLabels
from flash.vision.segmentation.transforms import default_train_transforms, default_val_transforms

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None


class SemanticSegmentationNumpyDataSource(NumpyDataSource):

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        sample[DefaultDataKeys.INPUT] = torch.from_numpy(sample[DefaultDataKeys.INPUT]).float()
        return sample


class SemanticSegmentationPathsDataSource(PathsDataSource):

    def __init__(self):
        super().__init__(IMG_EXTENSIONS)

    def load_data(self, data: Union[Tuple[str, str], Tuple[List[str], List[str]]],
                  dataset: BaseAutoDataset) -> Sequence[Mapping[str, Any]]:
        input_data, target_data = data

        if self.isdir(input_data) and self.isdir(target_data):
            input_files = os.listdir(input_data)
            target_files = os.listdir(target_data)

            all_files = set(input_files).intersection(set(target_files))

            if len(all_files) != len(input_files) or len(all_files) != len(target_files):
                rank_zero_warn(
                    f"Found inconsistent files in input_dir: {input_data} and target_dir: {target_data}. Some files"
                    " have been dropped.",
                    UserWarning,
                )

            input_data = [os.path.join(input_data, file) for file in all_files]
            target_data = [os.path.join(target_data, file) for file in all_files]

        if not isinstance(input_data, list) and not isinstance(target_data, list):
            input_data = [input_data]
            target_data = [target_data]

        if len(input_data) != len(target_data):
            raise MisconfigurationException(
                f"The number of input files ({len(input_data)}) and number of target files ({len(target_data)}) must be"
                " the same.",
            )

        data = filter(
            lambda sample: (
                has_file_allowed_extension(sample[0], self.extensions) and
                has_file_allowed_extension(sample[1], self.extensions)
            ),
            zip(input_data, target_data),
        )

        return [{DefaultDataKeys.INPUT: input, DefaultDataKeys.TARGET: target} for input, target in data]

    def predict_load_data(self, data: Union[str, List[str]]):
        return super().predict_load_data(data)

    def load_sample(self, sample: Mapping[str, Any]) -> Mapping[str, Union[torch.Tensor, torch.Size]]:
        # unpack data paths
        img_path = sample[DefaultDataKeys.INPUT]
        img_labels_path = sample[DefaultDataKeys.TARGET]

        # load images directly to torch tensors
        img: torch.Tensor = torchvision.io.read_image(img_path)  # CxHxW
        img_labels: torch.Tensor = torchvision.io.read_image(img_labels_path)  # CxHxW
        img_labels = img_labels[0]  # HxW

        return {
            DefaultDataKeys.INPUT: img.float(),
            DefaultDataKeys.TARGET: img_labels.float(),
            DefaultDataKeys.METADATA: img.shape,
        }

    def predict_load_sample(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        return {DefaultDataKeys.INPUT: torchvision.io.read_image(sample[DefaultDataKeys.INPUT]).float()}


class SemanticSegmentationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        num_classes: int = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
    ) -> None:
        """Preprocess pipeline for semantic segmentation tasks.

        Args:
            train_transform: Dictionary with the set of transforms to apply during training.
            val_transform: Dictionary with the set of transforms to apply during validation.
            test_transform: Dictionary with the set of transforms to apply during testing.
            predict_transform: Dictionary with the set of transforms to apply during prediction.
            image_size: A tuple with the expected output image size.
        """
        self.image_size = image_size
        self.num_classes = num_classes
        if num_classes:
            labels_map = labels_map or SegmentationLabels.create_random_labels_map(num_classes)

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.FILES: SemanticSegmentationPathsDataSource(),
                DefaultDataSources.FOLDERS: SemanticSegmentationPathsDataSource(),
                DefaultDataSources.TENSORS: TensorDataSource(),
                DefaultDataSources.NUMPY: SemanticSegmentationNumpyDataSource(),
            },
            default_data_source=DefaultDataSources.FILES,
        )

        if labels_map:
            self.set_state(ImageLabelsMap(labels_map))

        self.labels_map = labels_map

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms, "image_size": self.image_size,
            "num_classes": self.num_classes,
            "labels_map": self.labels_map
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    @property
    def default_train_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_train_transforms(self.image_size)

    @property
    def default_val_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_val_transforms(self.image_size)

    @property
    def default_test_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_val_transforms(self.image_size)

    @property
    def default_predict_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_val_transforms(self.image_size)


class SemanticSegmentationData(DataModule):
    """Data module for semantic segmentation tasks."""

    preprocess_cls = SemanticSegmentationPreprocess

    @staticmethod
    def configure_data_fetcher(
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> 'SegmentationMatplotlibVisualization':
        return SegmentationMatplotlibVisualization(labels_map=labels_map)

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @classmethod
    def from_data_source(
        cls,
        data_source: str,
        train_data: Any = None,
        val_data: Any = None,
        test_data: Any = None,
        predict_data: Any = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **preprocess_kwargs: Any,
    ) -> 'DataModule':

        if 'num_classes' not in preprocess_kwargs:
            raise MisconfigurationException("`num_classes` should be provided during instantiation.")

        num_classes = preprocess_kwargs["num_classes"]

        labels_map = getattr(preprocess_kwargs, "labels_map",
                             None) or SegmentationLabels.create_random_labels_map(num_classes)

        data_fetcher = data_fetcher or cls.configure_data_fetcher(labels_map)

        if flash._IS_TESTING:
            data_fetcher.block_viz_window = True

        dm = super(SemanticSegmentationData, cls).from_data_source(
            data_source=data_source,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            predict_data=predict_data,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            **preprocess_kwargs
        )

        dm.train_dataset.num_classes = num_classes
        return dm

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        train_target_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_target_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_target_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        **preprocess_kwargs,
    ) -> 'DataModule':
        """Creates a :class:`~flash.vision.segmentation.data.SemanticSegmentationData` object from the given data
        folders and corresponding target folders.

        Args:
            train_folder: The folder containing the train data.
            train_target_folder: The folder containing the train targets (targets must have the same file name as their
                corresponding inputs).
            val_folder: The folder containing the validation data.
            val_target_folder: The folder containing the validation targets (targets must have the same file name as
                their corresponding inputs).
            test_folder: The folder containing the test data.
            test_target_folder: The folder containing the test targets (targets must have the same file name as their
                corresponding inputs).
            predict_folder: The folder containing the predict data.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.data.data_module.DataModule`.
            preprocess: The :class:`~flash.data.data.Preprocess` to pass to the
                :class:`~flash.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls`` will be constructed
                and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.data.data_module.DataModule`.
            num_classes: Number of classes within the segmentation mask.
            labels_map: Mapping between a class_id and its corresponding color.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = SemanticSegmentationData.from_folders(
                train_folder="train_folder",
                train_target_folder="train_masks",
            )
        """
        return cls.from_data_source(
            DefaultDataSources.FOLDERS,
            (train_folder, train_target_folder),
            (val_folder, val_target_folder),
            (test_folder, test_target_folder),
            predict_folder,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            labels_map=labels_map,
            **preprocess_kwargs,
        )


class SegmentationMatplotlibVisualization(BaseVisualization):
    """Process and show the image batch and its associated label using matplotlib.
    """

    def __init__(self, labels_map: Dict[int, Tuple[int, int, int]]):
        super().__init__()

        self.max_cols: int = 4  # maximum number of columns we accept
        self.block_viz_window: bool = True  # parameter to allow user to block visualisation windows
        self.labels_map: Dict[int, Tuple[int, int, int]] = labels_map

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
            sample = data[i]
            if isinstance(sample, dict):
                image = sample[DefaultDataKeys.INPUT]
                label = sample[DefaultDataKeys.TARGET]
            elif isinstance(sample, tuple):
                image = sample[0]
                label = sample[1]
            else:
                raise TypeError(f"Unknown data type. Got: {type(data)}.")
            # convert images and labels to numpy and stack horizontally
            image_vis: np.ndarray = self._to_numpy(image.byte())
            label_tmp: torch.Tensor = SegmentationLabels.labels_to_image(label.squeeze().byte(), self.labels_map)
            label_vis: np.ndarray = self._to_numpy(label_tmp)
            img_vis = np.hstack((image_vis, label_vis))
            # send to visualiser
            ax.imshow(img_vis)
            ax.axis('off')
        plt.show(block=self.block_viz_window)

    def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_load_sample"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_post_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_post_tensor_transform"
        self._show_images_and_labels(samples, len(samples), win_title)

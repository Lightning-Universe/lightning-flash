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
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import flash
from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys, ImageLabelsMap, InputFormat
from flash.core.data.io.input_base import Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.process import Deserializer
from flash.core.data.utilities.paths import filter_valid_files, PATH_TYPE
from flash.core.data.utilities.samples import to_samples
from flash.core.data.utils import image_default_loader
from flash.core.integrations.fiftyone.utils import FiftyOneLabelUtilities
from flash.core.utilities.imports import (
    _FIFTYONE_AVAILABLE,
    _MATPLOTLIB_AVAILABLE,
    _TORCHVISION_AVAILABLE,
    Image,
    lazy_import,
    requires,
)
from flash.core.utilities.stages import RunningStage
from flash.image.data import ImageDeserializer, IMG_EXTENSIONS
from flash.image.segmentation.output import SegmentationLabelsOutput
from flash.image.segmentation.transforms import default_transforms, predict_default_transforms, train_default_transforms

SampleCollection = None
if _FIFTYONE_AVAILABLE:
    fo = lazy_import("fiftyone")
    if TYPE_CHECKING:
        from fiftyone.core.collections import SampleCollection
else:
    fo = None

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None

if _TORCHVISION_AVAILABLE:
    import torchvision
    import torchvision.transforms.functional as FT


class SemanticSegmentationInput(Input):
    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = sample[DataKeys.INPUT].float()
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = sample[DataKeys.TARGET].long()
        sample[DataKeys.METADATA] = {"size": sample[DataKeys.INPUT].shape[-2:]}
        return sample


class SemanticSegmentationTensorInput(Input):
    def load_data(self, tensor: Any, masks: Any = None) -> List[Dict[str, Any]]:
        return to_samples(tensor, masks)


class SemanticSegmentationNumpyInput(Input):
    def load_data(self, array: Any, masks: Any = None) -> List[Dict[str, Any]]:
        return to_samples(array, masks)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = torch.from_numpy(sample[DataKeys.INPUT])
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = torch.from_numpy(sample[DataKeys.TARGET])
        return super().load_sample(sample)


class SemanticSegmentationFilesInput(SemanticSegmentationInput):
    def load_data(
        self, files: Union[PATH_TYPE, List[PATH_TYPE]], mask_files: Optional[Union[PATH_TYPE, List[PATH_TYPE]]] = None
    ) -> List[Dict[str, Any]]:
        if mask_files is None:
            files = filter_valid_files(files, valid_extensions=IMG_EXTENSIONS)
        else:
            files, masks = filter_valid_files(files, mask_files, valid_extensions=IMG_EXTENSIONS)
        return to_samples(files, mask_files)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        sample[DataKeys.INPUT] = FT.to_tensor(image_default_loader(filepath))
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = torchvision.io.read_image(sample[DataKeys.TARGET])[0]
        sample = super().load_sample(sample)
        sample[DataKeys.METADATA]["filepath"] = filepath
        return sample


class SemanticSegmentationFolderInput(SemanticSegmentationFilesInput):
    def load_data(self, folder: PATH_TYPE, mask_folder: Optional[PATH_TYPE] = None) -> List[Dict[str, Any]]:
        files = os.listdir(folder)
        if mask_folder is not None:
            mask_files = os.listdir(mask_folder)

            all_files = set(files).intersection(set(mask_files))
            if len(all_files) != len(files) or len(all_files) != len(mask_files):
                rank_zero_warn(
                    f"Found inconsistent files in input folder: {folder} and mask folder: {mask_folder}. Some files"
                    " have been dropped.",
                    UserWarning,
                )

            files = [os.path.join(folder, file) for file in all_files]
            mask_files = [os.path.join(mask_folder, file) for file in all_files]
            return super().load_data(files, mask_files)
        return super().load_data(files)


class SemanticSegmentationFiftyOneInput(SemanticSegmentationFilesInput):
    def load_data(self, sample_collection: SampleCollection, label_field: str = "ground_truth") -> List[Dict[str, Any]]:
        label_utilities = FiftyOneLabelUtilities(label_field, fo.Segmentation)
        label_utilities.validate(sample_collection)
        self._fo_dataset_name = sample_collection.name
        return to_samples(sample_collection.values("filepath"))

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        sample = super().load_sample(sample)
        if not self.predicting:
            fo_dataset = fo.load_dataset(self._fo_dataset_name)
            fo_sample = fo_dataset[filepath]
            sample[DataKeys.TARGET] = torch.from_numpy(fo_sample[self.label_field].mask).long()  # H x W
        return sample


class SemanticSegmentationDeserializer(ImageDeserializer):
    def deserialize(self, data: str) -> Dict[str, Any]:
        result = super().deserialize(data)
        result[DataKeys.INPUT] = FT.to_tensor(result[DataKeys.INPUT])
        result[DataKeys.METADATA] = {"size": result[DataKeys.INPUT].shape}
        return result


class SemanticSegmentationInputTransform(InputTransform):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (128, 128),
        deserializer: Optional["Deserializer"] = None,
        num_classes: int = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
    ) -> None:
        """InputTransform pipeline for semantic segmentation tasks.

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
            labels_map = labels_map or SegmentationLabelsOutput.create_random_labels_map(num_classes)

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.FIFTYONE: SemanticSegmentationFiftyOneInput,
                InputFormat.FILES: SemanticSegmentationFilesInput,
                InputFormat.FOLDERS: SemanticSegmentationFolderInput,
                InputFormat.TENSORS: SemanticSegmentationTensorInput,
                InputFormat.NUMPY: SemanticSegmentationNumpyInput,
            },
            deserializer=deserializer or SemanticSegmentationDeserializer(),
            default_input=InputFormat.FILES,
        )

        if labels_map:
            self.set_state(ImageLabelsMap(labels_map))

        self.labels_map = labels_map

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "labels_map": self.labels_map,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms(self.image_size)

    def train_default_transforms(self) -> Optional[Dict[str, Callable]]:
        return train_default_transforms(self.image_size)

    def predict_default_transforms(self) -> Optional[Dict[str, Callable]]:
        return predict_default_transforms(self.image_size)


class SemanticSegmentationData(DataModule):
    """Data module for semantic segmentation tasks."""

    input_transform_cls = SemanticSegmentationInputTransform

    @staticmethod
    def configure_data_fetcher(
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> "SegmentationMatplotlibVisualization":
        return SegmentationMatplotlibVisualization(labels_map=labels_map)

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @classmethod
    def from_input(
        cls,
        input: str,
        train_data: Any = None,
        val_data: Any = None,
        test_data: Any = None,
        predict_data: Any = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        input_transform: Optional[InputTransform] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        **input_transform_kwargs: Any,
    ) -> "DataModule":

        if "num_classes" not in input_transform_kwargs:
            raise MisconfigurationException("`num_classes` should be provided during instantiation.")

        num_classes = input_transform_kwargs["num_classes"]

        labels_map = getattr(
            input_transform_kwargs, "labels_map", None
        ) or SegmentationLabelsOutput.create_random_labels_map(num_classes)

        data_fetcher = data_fetcher or cls.configure_data_fetcher(labels_map)

        if flash._IS_TESTING:
            data_fetcher.block_viz_window = True

        dm = super().from_input(
            input=input,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            predict_data=predict_data,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            input_transform=input_transform,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            **input_transform_kwargs,
        )

        if dm.train_dataset is not None:
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
        input_transform: Optional[InputTransform] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        **input_transform_kwargs,
    ) -> "DataModule":
        """Creates a :class:`~flash.image.segmentation.data.SemanticSegmentationData` object from the given data
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
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            input_transform: The :class:`~flash.core.data.data.InputTransform` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.input_transform_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_classes: Number of classes within the segmentation mask.
            labels_map: Mapping between a class_id and its corresponding color.
            input_transform_kwargs: Additional keyword arguments to use when constructing the input_transform.
                Will only be used if ``input_transform = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = SemanticSegmentationData.from_folders(
                train_folder="train_folder",
                train_target_folder="train_masks",
            )
        """
        return cls.from_input(
            InputFormat.FOLDERS,
            (train_folder, train_target_folder),
            (val_folder, val_target_folder),
            (test_folder, test_target_folder),
            predict_folder,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            input_transform=input_transform,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            labels_map=labels_map,
            **input_transform_kwargs,
        )


class SegmentationMatplotlibVisualization(BaseVisualization):
    """Process and show the image batch and its associated label using matplotlib."""

    def __init__(self, labels_map: Dict[int, Tuple[int, int, int]]):
        super().__init__()

        self.max_cols: int = 4  # maximum number of columns we accept
        self.block_viz_window: bool = True  # parameter to allow user to block visualisation windows
        self.labels_map: Dict[int, Tuple[int, int, int]] = labels_map

    @staticmethod
    @requires("image")
    def _to_numpy(img: Union[torch.Tensor, Image.Image]) -> np.ndarray:
        out: np.ndarray
        if isinstance(img, Image.Image):
            out = np.array(img)
        elif isinstance(img, torch.Tensor):
            out = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError(f"Unknown image type. Got: {type(img)}.")
        return out

    @requires("matplotlib")
    def _show_images_and_labels(self, data: List[Any], num_samples: int, title: str):
        # define the image grid
        cols: int = min(num_samples, self.max_cols)
        rows: int = num_samples // cols

        # create figure and set title
        fig, axs = plt.subplots(rows, cols)
        fig.suptitle(title)

        for i, ax in enumerate(axs.ravel()):
            # unpack images and labels
            sample = data[i]
            if isinstance(sample, dict):
                image = sample[DataKeys.INPUT]
                label = sample[DataKeys.TARGET]
            elif isinstance(sample, tuple):
                image = sample[0]
                label = sample[1]
            else:
                raise TypeError(f"Unknown data type. Got: {type(data)}.")
            # convert images and labels to numpy and stack horizontally
            image_vis: np.ndarray = self._to_numpy(image.byte())
            label_tmp: torch.Tensor = SegmentationLabelsOutput.labels_to_image(label.squeeze().byte(), self.labels_map)
            label_vis: np.ndarray = self._to_numpy(label_tmp)
            img_vis = np.hstack((image_vis, label_vis))
            # send to visualiser
            ax.imshow(img_vis)
            ax.axis("off")
        plt.show(block=self.block_viz_window)

    def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_load_sample"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_per_sample_transform(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_per_sample_transform"
        self._show_images_and_labels(samples, len(samples), win_title)

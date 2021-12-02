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
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn

from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
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
    def load_labels_map(
        self, num_classes: Optional[int] = None, labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> None:
        if num_classes is not None:
            self.num_classes = num_classes
            labels_map = labels_map or SegmentationLabelsOutput.create_random_labels_map(num_classes)

        if labels_map is not None:
            self.set_state(ImageLabelsMap(labels_map))
            self.labels_map = labels_map

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = sample[DataKeys.INPUT].float()
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = sample[DataKeys.TARGET].float()
        sample[DataKeys.METADATA] = {"size": sample[DataKeys.INPUT].shape[-2:]}
        return sample


class SemanticSegmentationTensorInput(SemanticSegmentationInput):
    def load_data(
        self,
        tensor: Any,
        masks: Any = None,
        num_classes: Optional[int] = None,
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        self.load_labels_map(num_classes, labels_map)
        return to_samples(tensor, masks)


class SemanticSegmentationNumpyInput(SemanticSegmentationInput):
    def load_data(
        self,
        array: Any,
        masks: Any = None,
        num_classes: Optional[int] = None,
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        self.load_labels_map(num_classes, labels_map)
        return to_samples(array, masks)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = torch.from_numpy(sample[DataKeys.INPUT])
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = torch.from_numpy(sample[DataKeys.TARGET])
        return super().load_sample(sample)


class SemanticSegmentationFilesInput(SemanticSegmentationInput):
    def load_data(
        self,
        files: Union[PATH_TYPE, List[PATH_TYPE]],
        mask_files: Optional[Union[PATH_TYPE, List[PATH_TYPE]]] = None,
        num_classes: Optional[int] = None,
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        self.load_labels_map(num_classes, labels_map)
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
    def load_data(
        self,
        folder: PATH_TYPE,
        mask_folder: Optional[PATH_TYPE] = None,
        num_classes: Optional[int] = None,
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        self.load_labels_map(num_classes, labels_map)
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
    def load_data(
        self,
        sample_collection: SampleCollection,
        label_field: str = "ground_truth",
        num_classes: Optional[int] = None,
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        self.load_labels_map(num_classes, labels_map)

        self.label_field = label_field
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
            sample[DataKeys.TARGET] = torch.from_numpy(fo_sample[self.label_field].mask).float()  # H x W
        return sample


class SemanticSegmentationDeserializer(ImageDeserializer):
    def deserialize(self, data: str) -> Dict[str, Any]:
        result = super().deserialize(data)
        result[DataKeys.INPUT] = FT.to_tensor(result[DataKeys.INPUT])
        result[DataKeys.METADATA] = {"size": result[DataKeys.INPUT].shape[-2:]}
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

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "image_size": self.image_size,
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

    @property
    def labels_map(self) -> Optional[Dict[int, Tuple[int, int, int]]]:
        return getattr(self.train_dataset, "labels_map", None)

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[str]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[str]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[str]] = None,
        predict_files: Optional[Sequence[str]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        image_size: Tuple[int, int] = (128, 128),
        **data_module_kwargs,
    ) -> "SemanticSegmentationData":
        dataset_kwargs = dict(num_classes=num_classes, labels_map=labels_map, data_pipeline_state=DataPipelineState())

        return cls(
            SemanticSegmentationFilesInput(RunningStage.TRAINING, train_files, train_targets, **dataset_kwargs),
            SemanticSegmentationFilesInput(RunningStage.VALIDATING, val_files, val_targets, **dataset_kwargs),
            SemanticSegmentationFilesInput(RunningStage.TESTING, test_files, test_targets, **dataset_kwargs),
            SemanticSegmentationFilesInput(RunningStage.PREDICTING, predict_files, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                image_size=image_size,
            ),
            **data_module_kwargs,
        )

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
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        image_size: Tuple[int, int] = (128, 128),
        **data_module_kwargs,
    ) -> "SemanticSegmentationData":
        dataset_kwargs = dict(num_classes=num_classes, labels_map=labels_map, data_pipeline_state=DataPipelineState())

        return cls(
            SemanticSegmentationFolderInput(RunningStage.TRAINING, train_folder, train_target_folder, **dataset_kwargs),
            SemanticSegmentationFolderInput(RunningStage.VALIDATING, val_folder, val_target_folder, **dataset_kwargs),
            SemanticSegmentationFolderInput(RunningStage.TESTING, test_folder, test_target_folder, **dataset_kwargs),
            SemanticSegmentationFolderInput(RunningStage.PREDICTING, predict_folder, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                image_size=image_size,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        train_targets: Optional[Collection[np.ndarray]] = None,
        val_data: Optional[Collection[np.ndarray]] = None,
        val_targets: Optional[Sequence[np.ndarray]] = None,
        test_data: Optional[Collection[np.ndarray]] = None,
        test_targets: Optional[Sequence[np.ndarray]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        image_size: Tuple[int, int] = (128, 128),
        **data_module_kwargs,
    ) -> "SemanticSegmentationData":
        dataset_kwargs = dict(num_classes=num_classes, labels_map=labels_map, data_pipeline_state=DataPipelineState())

        return cls(
            SemanticSegmentationNumpyInput(RunningStage.TRAINING, train_data, train_targets, **dataset_kwargs),
            SemanticSegmentationNumpyInput(RunningStage.VALIDATING, val_data, val_targets, **dataset_kwargs),
            SemanticSegmentationNumpyInput(RunningStage.TESTING, test_data, test_targets, **dataset_kwargs),
            SemanticSegmentationNumpyInput(RunningStage.PREDICTING, predict_data, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                image_size=image_size,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Collection[torch.Tensor]] = None,
        train_targets: Optional[Collection[torch.Tensor]] = None,
        val_data: Optional[Collection[torch.Tensor]] = None,
        val_targets: Optional[Sequence[torch.Tensor]] = None,
        test_data: Optional[Collection[torch.Tensor]] = None,
        test_targets: Optional[Sequence[torch.Tensor]] = None,
        predict_data: Optional[Collection[torch.Tensor]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        image_size: Tuple[int, int] = (128, 128),
        **data_module_kwargs,
    ) -> "SemanticSegmentationData":
        dataset_kwargs = dict(num_classes=num_classes, labels_map=labels_map, data_pipeline_state=DataPipelineState())

        return cls(
            SemanticSegmentationTensorInput(RunningStage.TRAINING, train_data, train_targets, **dataset_kwargs),
            SemanticSegmentationTensorInput(RunningStage.VALIDATING, val_data, val_targets, **dataset_kwargs),
            SemanticSegmentationTensorInput(RunningStage.TESTING, test_data, test_targets, **dataset_kwargs),
            SemanticSegmentationTensorInput(RunningStage.PREDICTING, predict_data, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                image_size=image_size,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_fiftyone(
        cls,
        train_dataset: Optional[SampleCollection] = None,
        val_dataset: Optional[SampleCollection] = None,
        test_dataset: Optional[SampleCollection] = None,
        predict_dataset: Optional[SampleCollection] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        label_field: str = "ground_truth",
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        image_size: Tuple[int, int] = (128, 128),
        **data_module_kwargs: Any,
    ) -> "SemanticSegmentationData":
        dataset_kwargs = dict(
            label_field=label_field,
            num_classes=num_classes,
            labels_map=labels_map,
            data_pipeline_state=DataPipelineState(),
        )

        return cls(
            SemanticSegmentationFiftyOneInput(RunningStage.TRAINING, train_dataset, **dataset_kwargs),
            SemanticSegmentationFiftyOneInput(RunningStage.VALIDATING, val_dataset, **dataset_kwargs),
            SemanticSegmentationFiftyOneInput(RunningStage.TESTING, test_dataset, **dataset_kwargs),
            SemanticSegmentationFiftyOneInput(RunningStage.PREDICTING, predict_dataset, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                image_size=image_size,
            ),
            **data_module_kwargs,
        )

    def configure_data_fetcher(self) -> BaseDataFetcher:
        return SegmentationMatplotlibVisualization(labels_map=self.labels_map)

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value


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

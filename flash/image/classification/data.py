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
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.classification_input import ClassificationInput, ClassificationState
from flash.core.data.io.input import DataKeys, InputFormat
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.process import Deserializer
from flash.core.data.utilities.classification import TargetMode
from flash.core.data.utilities.data_frame import read_csv, resolve_files, resolve_targets
from flash.core.data.utilities.paths import filter_valid_files, make_dataset, PATH_TYPE
from flash.core.data.utilities.samples import to_samples
from flash.core.integrations.fiftyone.utils import FiftyOneLabelUtilities
from flash.core.integrations.labelstudio.input import _parse_labelstudio_arguments, LabelStudioImageClassificationInput
from flash.core.utilities.imports import _MATPLOTLIB_AVAILABLE, Image, requires
from flash.core.utilities.stages import RunningStage
from flash.image.classification.transforms import default_transforms, train_default_transforms
from flash.image.data import (
    fol,
    ImageDeserializer,
    ImageFilesInput,
    ImageNumpyInput,
    ImageTensorInput,
    IMG_EXTENSIONS,
    NP_EXTENSIONS,
    SampleCollection,
)

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None


class ImageClassificationFilesInput(ClassificationInput, ImageFilesInput):
    def load_data(
        self,
        files: List[PATH_TYPE],
        targets: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        if targets is None:
            return super().load_data(files)
        files, targets = filter_valid_files(files, targets, valid_extensions=IMG_EXTENSIONS + NP_EXTENSIONS)
        self.load_target_metadata(targets)
        return to_samples(files, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample


class ImageClassificationFolderInput(ImageClassificationFilesInput):
    def load_data(self, folder: PATH_TYPE) -> List[Dict[str, Any]]:
        files, targets = make_dataset(folder, extensions=IMG_EXTENSIONS + NP_EXTENSIONS)
        return super().load_data(files, targets)


class ImageClassificationFiftyOneInput(ImageClassificationFilesInput):
    @requires("fiftyone")
    def load_data(self, sample_collection: SampleCollection, label_field: str = "ground_truth") -> List[Dict[str, Any]]:
        label_utilities = FiftyOneLabelUtilities(label_field, fol.Label)
        label_utilities.validate(sample_collection)

        label_path = sample_collection._get_label_field_path(label_field, "label")[1]

        filepaths = sample_collection.values("filepath")
        targets = sample_collection.values(label_path)

        return super().load_data(filepaths, targets)

    @staticmethod
    @requires("fiftyone")
    def predict_load_data(data: SampleCollection) -> List[Dict[str, Any]]:
        return super().load_data(data.values("filepath"))


class ImageClassificationTensorInput(ClassificationInput, ImageTensorInput):
    def load_data(self, tensor: Any, targets: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        if targets is not None:
            self.load_target_metadata(targets)
        return to_samples(tensor, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample


class ImageClassificationNumpyInput(ClassificationInput, ImageNumpyInput):
    def load_data(self, array: Any, targets: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        if targets is not None:
            self.load_target_metadata(targets)
        return to_samples(array, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample


class ImageClassificationDataFrameInput(ImageClassificationFilesInput):
    def load_data(
        self,
        data_frame: pd.DataFrame,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        root: Optional[PATH_TYPE] = None,
        resolver: Optional[Callable[[Optional[PATH_TYPE], Any], PATH_TYPE]] = None,
    ) -> List[Dict[str, Any]]:
        files = resolve_files(data_frame, input_key, root, resolver)
        if target_keys is not None:
            targets = resolve_targets(data_frame, target_keys)
        else:
            targets = None
        result = super().load_data(files, targets)

        # If we had binary multi-class targets then we also know the labels (column names)
        if self.training and self.target_mode is TargetMode.MULTI_BINARY and isinstance(target_keys, List):
            classification_state = self.get_state(ClassificationState)
            self.set_state(ClassificationState(target_keys, classification_state.num_classes))

        return result


class ImageClassificationCSVInput(ImageClassificationDataFrameInput):
    def load_data(
        self,
        csv_file: PATH_TYPE,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        root: Optional[PATH_TYPE] = None,
        resolver: Optional[Callable[[Optional[PATH_TYPE], Any], PATH_TYPE]] = None,
    ) -> List[Dict[str, Any]]:
        data_frame = read_csv(csv_file)
        if root is None:
            root = os.path.dirname(csv_file)
        return super().load_data(data_frame, input_key, target_keys, root, resolver)


class ImageClassificationInputTransform(InputTransform):
    """Preprocssing of data of image classification."""

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        deserializer: Optional[Deserializer] = None,
    ):
        self.image_size = image_size

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.FIFTYONE: ImageClassificationFiftyOneInput,
                InputFormat.FILES: ImageClassificationFilesInput,
                InputFormat.FOLDERS: ImageClassificationFolderInput,
                InputFormat.NUMPY: ImageClassificationNumpyInput,
                InputFormat.TENSORS: ImageClassificationTensorInput,
                InputFormat.DATAFRAME: ImageClassificationDataFrameInput,
                InputFormat.CSV: ImageClassificationCSVInput,
                InputFormat.LABELSTUDIO: LabelStudioImageClassificationInput,
            },
            deserializer=deserializer or ImageDeserializer(),
            default_input=InputFormat.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms, "image_size": self.image_size}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms(self.image_size)

    def train_default_transforms(self) -> Optional[Dict[str, Callable]]:
        return train_default_transforms(self.image_size)


class ImageClassificationData(DataModule):
    """Data module for image classification tasks."""

    input_transform_cls = ImageClassificationInputTransform

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[Any]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_files: Optional[Sequence[str]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":

        dataset_kwargs = dict(data_pipeline_state=DataPipelineState())

        return cls(
            ImageClassificationFilesInput(RunningStage.TRAINING, train_files, train_targets, **dataset_kwargs),
            ImageClassificationFilesInput(RunningStage.VALIDATING, val_files, val_targets, **dataset_kwargs),
            ImageClassificationFilesInput(RunningStage.TESTING, test_files, test_targets, **dataset_kwargs),
            ImageClassificationFilesInput(RunningStage.PREDICTING, predict_files, **dataset_kwargs),
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
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":

        dataset_kwargs = dict(data_pipeline_state=DataPipelineState())

        return cls(
            ImageClassificationFolderInput(RunningStage.TRAINING, train_folder, **dataset_kwargs),
            ImageClassificationFolderInput(RunningStage.VALIDATING, val_folder, **dataset_kwargs),
            ImageClassificationFolderInput(RunningStage.TESTING, test_folder, **dataset_kwargs),
            ImageClassificationFolderInput(RunningStage.PREDICTING, predict_folder, **dataset_kwargs),
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
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[np.ndarray]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[np.ndarray]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":

        dataset_kwargs = dict(data_pipeline_state=DataPipelineState())

        return cls(
            ImageClassificationNumpyInput(RunningStage.TRAINING, train_data, train_targets, **dataset_kwargs),
            ImageClassificationNumpyInput(RunningStage.VALIDATING, val_data, val_targets, **dataset_kwargs),
            ImageClassificationNumpyInput(RunningStage.TESTING, test_data, test_targets, **dataset_kwargs),
            ImageClassificationNumpyInput(RunningStage.PREDICTING, predict_data, **dataset_kwargs),
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
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[torch.Tensor]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[torch.Tensor]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[torch.Tensor]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":

        dataset_kwargs = dict(data_pipeline_state=DataPipelineState())

        return cls(
            ImageClassificationTensorInput(RunningStage.TRAINING, train_data, train_targets, **dataset_kwargs),
            ImageClassificationTensorInput(RunningStage.VALIDATING, val_data, val_targets, **dataset_kwargs),
            ImageClassificationTensorInput(RunningStage.TESTING, test_data, test_targets, **dataset_kwargs),
            ImageClassificationTensorInput(RunningStage.PREDICTING, predict_data, **dataset_kwargs),
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
    def from_data_frame(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_data_frame: Optional[pd.DataFrame] = None,
        train_images_root: Optional[str] = None,
        train_resolver: Optional[Callable[[str, str], str]] = None,
        val_data_frame: Optional[pd.DataFrame] = None,
        val_images_root: Optional[str] = None,
        val_resolver: Optional[Callable[[str, str], str]] = None,
        test_data_frame: Optional[pd.DataFrame] = None,
        test_images_root: Optional[str] = None,
        test_resolver: Optional[Callable[[str, str], str]] = None,
        predict_data_frame: Optional[pd.DataFrame] = None,
        predict_images_root: Optional[str] = None,
        predict_resolver: Optional[Callable[[str, str], str]] = None,
        train_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":

        dataset_kwargs = dict(data_pipeline_state=DataPipelineState())

        train_data = (train_data_frame, input_field, target_fields, train_images_root, train_resolver)
        val_data = (val_data_frame, input_field, target_fields, val_images_root, val_resolver)
        test_data = (test_data_frame, input_field, target_fields, test_images_root, test_resolver)
        predict_data = (predict_data_frame, input_field, predict_images_root, predict_resolver)

        return cls(
            ImageClassificationCSVInput(RunningStage.TRAINING, *train_data, **dataset_kwargs),
            ImageClassificationCSVInput(RunningStage.VALIDATING, *val_data, **dataset_kwargs),
            ImageClassificationCSVInput(RunningStage.TESTING, *test_data, **dataset_kwargs),
            ImageClassificationCSVInput(RunningStage.PREDICTING, *predict_data, **dataset_kwargs),
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
    def from_csv(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, List[str]]] = None,
        train_file: Optional[PATH_TYPE] = None,
        train_images_root: Optional[PATH_TYPE] = None,
        train_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        val_file: Optional[PATH_TYPE] = None,
        val_images_root: Optional[PATH_TYPE] = None,
        val_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        test_file: Optional[str] = None,
        test_images_root: Optional[str] = None,
        test_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        predict_file: Optional[str] = None,
        predict_images_root: Optional[str] = None,
        predict_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        train_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":

        dataset_kwargs = dict(data_pipeline_state=DataPipelineState())

        train_data = (train_file, input_field, target_fields, train_images_root, train_resolver)
        val_data = (val_file, input_field, target_fields, val_images_root, val_resolver)
        test_data = (test_file, input_field, target_fields, test_images_root, test_resolver)
        predict_data = (predict_file, input_field, predict_images_root, predict_resolver)

        return cls(
            ImageClassificationCSVInput(RunningStage.TRAINING, *train_data, **dataset_kwargs),
            ImageClassificationCSVInput(RunningStage.VALIDATING, *val_data, **dataset_kwargs),
            ImageClassificationCSVInput(RunningStage.TESTING, *test_data, **dataset_kwargs),
            ImageClassificationCSVInput(RunningStage.PREDICTING, *predict_data, **dataset_kwargs),
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
    @requires("fiftyone")
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
        image_size: Tuple[int, int] = (196, 196),
        **data_module_kwargs,
    ) -> "ImageClassificationData":

        dataset_kwargs = dict(data_pipeline_state=DataPipelineState())

        return cls(
            ImageClassificationFiftyOneInput(RunningStage.TRAINING, train_dataset, label_field, **dataset_kwargs),
            ImageClassificationFiftyOneInput(RunningStage.VALIDATING, val_dataset, label_field, **dataset_kwargs),
            ImageClassificationFiftyOneInput(RunningStage.TESTING, test_dataset, label_field, **dataset_kwargs),
            ImageClassificationFiftyOneInput(RunningStage.PREDICTING, predict_dataset, label_field, **dataset_kwargs),
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
    def from_labelstudio(
        cls,
        export_json: str = None,
        train_export_json: str = None,
        val_export_json: str = None,
        test_export_json: str = None,
        predict_export_json: str = None,
        data_folder: str = None,
        train_data_folder: str = None,
        val_data_folder: str = None,
        test_data_folder: str = None,
        predict_data_folder: str = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        val_split: Optional[float] = None,
        multi_label: Optional[bool] = False,
        image_size: Tuple[int, int] = (196, 196),
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object
        from the given export file and data directory using the
        :class:`~flash.core.data.io.input.Input` of name
        :attr:`~flash.core.data.io.input.InputFormat.FOLDERS`
        from the passed or constructed :class:`~flash.core.data.io.input_transform.InputTransform`.

        Args:
            export_json: path to label studio export file
            train_export_json: path to label studio export file for train set,
            overrides export_json if specified
            val_export_json: path to label studio export file for validation
            test_export_json: path to label studio export file for test
            predict_export_json: path to label studio export file for predict
            data_folder: path to label studio data folder
            train_data_folder: path to label studio data folder for train data set,
            overrides data_folder if specified
            val_data_folder: path to label studio data folder for validation data
            test_data_folder: path to label studio data folder for test data
            predict_data_folder: path to label studio data folder for predict data
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
            input_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.input_transform_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            multi_label: Whether the labels are multi encoded
            image_size: Size of the image.
            data_module_kwargs: Additional keyword arguments to use when constructing the datamodule.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_labelstudio(
                export_json='project.json',
                data_folder='label-studio/media/upload',
                val_split=0.8,
            )
        """

        train_data, val_data, test_data, predict_data = _parse_labelstudio_arguments(
            export_json=export_json,
            train_export_json=train_export_json,
            val_export_json=val_export_json,
            test_export_json=test_export_json,
            predict_export_json=predict_export_json,
            data_folder=data_folder,
            train_data_folder=train_data_folder,
            val_data_folder=val_data_folder,
            test_data_folder=test_data_folder,
            predict_data_folder=predict_data_folder,
            val_split=val_split,
            multi_label=multi_label,
        )

        dataset_kwargs = dict(data_pipeline_state=DataPipelineState())

        return cls(
            LabelStudioImageClassificationInput(RunningStage.TRAINING, train_data, **dataset_kwargs),
            LabelStudioImageClassificationInput(RunningStage.VALIDATING, val_data, **dataset_kwargs),
            LabelStudioImageClassificationInput(RunningStage.TESTING, test_data, **dataset_kwargs),
            LabelStudioImageClassificationInput(RunningStage.PREDICTING, predict_data, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                image_size=image_size,
            ),
            **data_module_kwargs,
        )

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return MatplotlibVisualization(*args, **kwargs)


class MatplotlibVisualization(BaseVisualization):
    """Process and show the image batch and its associated label using matplotlib."""

    max_cols: int = 4  # maximum number of columns we accept
    block_viz_window: bool = True  # parameter to allow user to block visualisation windows

    @staticmethod
    @requires("image")
    def _to_numpy(img: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
        out: np.ndarray
        if isinstance(img, np.ndarray):
            out = img
        elif isinstance(img, Image.Image):
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

        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for i, ax in enumerate(axs):
            # unpack images and labels
            if isinstance(data, list):
                _img, _label = data[i][DataKeys.INPUT], data[i].get(DataKeys.TARGET, "")
            elif isinstance(data, dict):
                _img, _label = data[DataKeys.INPUT][i], data.get(DataKeys.TARGET, [""] * (i + 1))[i]
            else:
                raise TypeError(f"Unknown data type. Got: {type(data)}.")
            # convert images to numpy
            _img: np.ndarray = self._to_numpy(_img)
            if isinstance(_label, torch.Tensor):
                _label = _label.squeeze().tolist()
            # show image and set label as subplot title
            ax.imshow(_img)
            ax.set_title(str(_label))
            ax.axis("off")
        plt.show(block=self.block_viz_window)

    def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_load_sample"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_per_sample_transform(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_per_sample_transform"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_per_batch_transform(self, batch: List[Any], running_stage):
        win_title: str = f"{running_stage} - show_per_batch_transform"
        self._show_images_and_labels(batch[0], batch[0][DataKeys.INPUT].shape[0], win_title)

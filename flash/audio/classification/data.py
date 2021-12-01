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

from flash.audio.classification.transforms import default_transforms, train_default_transforms
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.io.classification_input import ClassificationInput, ClassificationState
from flash.core.data.io.input import DataKeys, has_file_allowed_extension, InputFormat
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.process import Deserializer
from flash.core.data.utilities.classification import TargetMode
from flash.core.data.utilities.data_frame import read_csv, resolve_files, resolve_targets
from flash.core.data.utilities.paths import filter_valid_files, make_dataset, PATH_TYPE
from flash.core.data.utilities.samples import to_samples
from flash.core.data.utils import image_default_loader
from flash.core.utilities.imports import requires
from flash.core.utilities.stages import RunningStage
from flash.image.classification.data import MatplotlibVisualization
from flash.image.data import ImageDeserializer, IMG_EXTENSIONS, NP_EXTENSIONS


def spectrogram_loader(filepath: str):
    if has_file_allowed_extension(filepath, IMG_EXTENSIONS):
        img = image_default_loader(filepath)
        data = np.array(img)
    else:
        data = np.load(filepath)
    return data


class AudioClassificationInput(ClassificationInput):
    @requires("audio")
    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        h, w = sample[DataKeys.INPUT].shape[-2:]  # H x W
        if DataKeys.METADATA not in sample:
            sample[DataKeys.METADATA] = {}
        sample[DataKeys.METADATA]["size"] = (h, w)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample


class AudioClassificationFilesInput(AudioClassificationInput):
    def load_data(
        self,
        files: List[PATH_TYPE],
        targets: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        if targets is None:
            files = filter_valid_files(files, valid_extensions=IMG_EXTENSIONS + NP_EXTENSIONS)
            return to_samples(files)
        files, targets = filter_valid_files(files, targets, valid_extensions=IMG_EXTENSIONS + NP_EXTENSIONS)
        self.load_target_metadata(targets)
        return to_samples(files, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        sample[DataKeys.INPUT] = spectrogram_loader(filepath)
        sample = super().load_sample(sample)
        sample[DataKeys.METADATA]["filepath"] = filepath
        return sample


class AudioClassificationFolderInput(AudioClassificationFilesInput):
    def load_data(self, folder: PATH_TYPE) -> List[Dict[str, Any]]:
        files, targets = make_dataset(folder, extensions=IMG_EXTENSIONS + NP_EXTENSIONS)
        return super().load_data(files, targets)


class AudioClassificationNumpyInput(AudioClassificationInput):
    def load_data(self, array: Any, targets: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        if targets is not None:
            self.load_target_metadata(targets)
        return to_samples(array, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = np.transpose(sample[DataKeys.INPUT], (1, 2, 0))
        return sample


class AudioClassificationTensorInput(AudioClassificationNumpyInput):
    def load_data(self, tensor: Any, targets: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        if targets is not None:
            self.load_target_metadata(targets)
        return to_samples(tensor, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = sample[DataKeys.INPUT].numpy()
        return sample


class AudioClassificationDataFrameInput(AudioClassificationFilesInput):
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


class AudioClassificationCSVInput(AudioClassificationDataFrameInput):
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


class AudioClassificationInputTransform(InputTransform):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        spectrogram_size: Tuple[int, int] = (128, 128),
        time_mask_param: Optional[int] = None,
        freq_mask_param: Optional[int] = None,
        deserializer: Optional["Deserializer"] = None,
    ):
        self.spectrogram_size = spectrogram_size
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.FILES: AudioClassificationFilesInput,
                InputFormat.FOLDERS: AudioClassificationFolderInput,
                InputFormat.DATAFRAME: AudioClassificationDataFrameInput,
                InputFormat.CSV: AudioClassificationDataFrameInput,
                InputFormat.NUMPY: AudioClassificationNumpyInput,
                InputFormat.TENSORS: AudioClassificationTensorInput,
            },
            deserializer=deserializer or ImageDeserializer(),
            default_input=InputFormat.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "spectrogram_size": self.spectrogram_size,
            "time_mask_param": self.time_mask_param,
            "freq_mask_param": self.freq_mask_param,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms(self.spectrogram_size)

    def train_default_transforms(self) -> Optional[Dict[str, Callable]]:
        return train_default_transforms(self.spectrogram_size, self.time_mask_param, self.freq_mask_param)


class AudioClassificationData(DataModule):
    """Data module for audio classification."""

    input_transform_cls = AudioClassificationInputTransform

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
        spectrogram_size: Tuple[int, int] = (128, 128),
        time_mask_param: Optional[int] = None,
        freq_mask_param: Optional[int] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        return cls(
            AudioClassificationFilesInput(RunningStage.TRAINING, train_files, train_targets),
            AudioClassificationFilesInput(RunningStage.VALIDATING, val_files, val_targets),
            AudioClassificationFilesInput(RunningStage.TESTING, test_files, test_targets),
            AudioClassificationFilesInput(RunningStage.PREDICTING, predict_files),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                spectrogram_size=spectrogram_size,
                time_mask_param=time_mask_param,
                freq_mask_param=freq_mask_param,
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
        spectrogram_size: Tuple[int, int] = (128, 128),
        time_mask_param: Optional[int] = None,
        freq_mask_param: Optional[int] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        return cls(
            AudioClassificationFolderInput(RunningStage.TRAINING, train_folder),
            AudioClassificationFolderInput(RunningStage.VALIDATING, val_folder),
            AudioClassificationFolderInput(RunningStage.TESTING, test_folder),
            AudioClassificationFolderInput(RunningStage.PREDICTING, predict_folder),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                spectrogram_size=spectrogram_size,
                time_mask_param=time_mask_param,
                freq_mask_param=freq_mask_param,
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
        spectrogram_size: Tuple[int, int] = (128, 128),
        time_mask_param: Optional[int] = None,
        freq_mask_param: Optional[int] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        return cls(
            AudioClassificationNumpyInput(RunningStage.TRAINING, train_data, train_targets),
            AudioClassificationNumpyInput(RunningStage.VALIDATING, val_data, val_targets),
            AudioClassificationNumpyInput(RunningStage.TESTING, test_data, test_targets),
            AudioClassificationNumpyInput(RunningStage.PREDICTING, predict_data),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                spectrogram_size=spectrogram_size,
                time_mask_param=time_mask_param,
                freq_mask_param=freq_mask_param,
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
        spectrogram_size: Tuple[int, int] = (128, 128),
        time_mask_param: Optional[int] = None,
        freq_mask_param: Optional[int] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        return cls(
            AudioClassificationTensorInput(RunningStage.TRAINING, train_data, train_targets),
            AudioClassificationTensorInput(RunningStage.VALIDATING, val_data, val_targets),
            AudioClassificationTensorInput(RunningStage.TESTING, test_data, test_targets),
            AudioClassificationTensorInput(RunningStage.PREDICTING, predict_data),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                spectrogram_size=spectrogram_size,
                time_mask_param=time_mask_param,
                freq_mask_param=freq_mask_param,
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
        spectrogram_size: Tuple[int, int] = (128, 128),
        time_mask_param: Optional[int] = None,
        freq_mask_param: Optional[int] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        return cls(
            AudioClassificationDataFrameInput(
                RunningStage.TRAINING, train_data_frame, input_field, target_fields, train_images_root, train_resolver
            ),
            AudioClassificationDataFrameInput(
                RunningStage.VALIDATING, val_data_frame, input_field, target_fields, val_images_root, val_resolver
            ),
            AudioClassificationDataFrameInput(
                RunningStage.TESTING, test_data_frame, input_field, target_fields, test_images_root, test_resolver
            ),
            AudioClassificationDataFrameInput(
                RunningStage.PREDICTING,
                predict_data_frame,
                input_field,
                root=predict_images_root,
                resolver=predict_resolver,
            ),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                spectrogram_size=spectrogram_size,
                time_mask_param=time_mask_param,
                freq_mask_param=freq_mask_param,
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
        spectrogram_size: Tuple[int, int] = (128, 128),
        time_mask_param: Optional[int] = None,
        freq_mask_param: Optional[int] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        return cls(
            AudioClassificationCSVInput(
                RunningStage.TRAINING, train_file, input_field, target_fields, train_images_root, train_resolver
            ),
            AudioClassificationCSVInput(
                RunningStage.VALIDATING, val_file, input_field, target_fields, val_images_root, val_resolver
            ),
            AudioClassificationCSVInput(
                RunningStage.TESTING, test_file, input_field, target_fields, test_images_root, test_resolver
            ),
            AudioClassificationCSVInput(
                RunningStage.PREDICTING, predict_file, input_field, root=predict_images_root, resolver=predict_resolver
            ),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                spectrogram_size=spectrogram_size,
                time_mask_param=time_mask_param,
                freq_mask_param=freq_mask_param,
            ),
            **data_module_kwargs,
        )

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return MatplotlibVisualization(*args, **kwargs)

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
from typing import Any, Optional

from flash.core.data.callback import BaseDataFetcher
from flash.core.data_v2.data_module import DataModule
from flash.core.data_v2.io.input import InputFormat
from flash.core.data_v2.transforms.input_transform import INPUT_TRANSFORM_TYPE
from flash.image.classification.data import MatplotlibVisualization
from flash.image.classification.transforms import IMAGE_CLASSIFICATION_INPUT_TRANSFORMS_REGISTRY
from flash.image.io import ImagePathsInput


class ImageClassificationDataModule(DataModule):
    """Data module for image classification tasks."""

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[INPUT_TRANSFORM_TYPE] = InputFormat.DEFAULT,
        val_transform: Optional[INPUT_TRANSFORM_TYPE] = InputFormat.DEFAULT,
        test_transform: Optional[INPUT_TRANSFORM_TYPE] = InputFormat.DEFAULT,
        predict_transform: Optional[INPUT_TRANSFORM_TYPE] = InputFormat.DEFAULT,
        **data_module_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data_v2.data_module.DataModule` object from the given folders using the
        :class:`~flash.core.data.data_source.DataSource` of name
        :attr:`~flash.core.data_v2.io.input.InputFormat.FOLDERS`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            train_folder: The folder containing the train data.
            val_folder: The folder containing the validation data.
            test_folder: The folder containing the test data.
            predict_folder: The folder containing the predict data.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_module_kwargs: The keywords arguments for the :class:`~flash.core.data_v2.data_module.DataModule`.

        Returns:
            The constructed data module.
        """
        return cls(
            *cls.create_inputs(
                ImagePathsInput,
                train_folder,
                val_folder,
                test_folder,
                predict_folder,
                train_transform=train_transform,
                val_transform=val_transform,
                test_transform=test_transform,
                predict_transform=predict_transform,
            ),
            **data_module_kwargs,
        )

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return MatplotlibVisualization(*args, **kwargs)


input_transforms_registry = IMAGE_CLASSIFICATION_INPUT_TRANSFORMS_REGISTRY
ImageClassificationDataModule.register_flash_dataset(InputFormat.DEFAULT, ImagePathsInput, input_transforms_registry)
ImageClassificationDataModule.register_flash_dataset(InputFormat.FOLDERS, ImagePathsInput, input_transforms_registry)

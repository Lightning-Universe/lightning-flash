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
from flash.core.data.data_source import DefaultDataSources
from flash.core.data_v2.data_module import DataModule
from flash.core.data_v2.transforms.input_transform import InputTransform
from flash.image.classification.data import MatplotlibVisualization
from flash.image.classification.transforms import IMAGE_CLASSIFICATION_REGISTRY
from flash.image.io import ImagePathsInput


class ImageClassificationData(DataModule):
    """Data module for image classification tasks."""

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[InputTransform] = "default",
        val_transform: Optional[InputTransform] = "default",
        test_transform: Optional[InputTransform] = "default",
        predict_transform: Optional[InputTransform] = "default",
        **data_module_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given folders using the
        :class:`~flash.core.data.data_source.DataSource` of name
        :attr:`~flash.core.data.data_source.DefaultDataSources.FOLDERS`
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
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.
        """
        return cls(
            *cls.create_flash_datasets(
                DefaultDataSources.FOLDERS,
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


ImagePathsInput.input_transforms_registry = IMAGE_CLASSIFICATION_REGISTRY
ImageClassificationData.register_flash_dataset(DefaultDataSources.FOLDERS, ImagePathsInput)

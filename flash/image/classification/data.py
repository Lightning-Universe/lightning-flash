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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import pandas as pd
from torch.utils.data.sampler import Sampler

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DefaultDataSources
from flash.core.data.process import Deserializer, Preprocess
from flash.image.classification.data_sources import ImageClassificationDataSources, ImageClassificationDefaultDataSource
from flash.image.classification.transforms import default_transforms, train_default_transforms
from flash.image.classification.viz import MatplotlibVisualization
from flash.image.data_sources import ImageDeserializer


class ImageClassificationPreprocess(Preprocess):

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
            deserializer=deserializer or ImageDeserializer(),
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

    preprocess_cls = ImageClassificationPreprocess
    data_sources = ImageClassificationDataSources
    default_data_source = ImageClassificationDefaultDataSource

    @classmethod
    def from_data_frame(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_data_frame: Optional[pd.DataFrame] = None,
        train_images_root: Optional[str] = None,
        val_data_frame: Optional[pd.DataFrame] = None,
        val_images_root: Optional[str] = None,
        test_data_frame: Optional[pd.DataFrame] = None,
        test_images_root: Optional[str] = None,
        predict_data_frame: Optional[pd.DataFrame] = None,
        predict_images_root: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        **preprocess_kwargs: Any,
    ) -> 'DataModule':
        """Creates a :class:`~flash.image.classification.data.ImageClassificationData` object from the given pandas
        ``DataFrame`` objects.

        Args:
            input_field: The field (column) in the pandas ``DataFrame`` to use for the input.
            target_fields: The field or fields (columns) in the pandas ``DataFrame`` to use for the target.
            train_data_frame: The pandas ``DataFrame`` containing the training data.
            train_images_root: The directory containing the train images. If ``None``, values in the ``input_field``
                will be assumed to be the full file paths.
            val_data_frame: The pandas ``DataFrame`` containing the validation data.
            val_images_root: The directory containing the validation images. If ``None``, the directory containing the
                ``val_file`` will be used.
            test_data_frame: The pandas ``DataFrame`` containing the testing data.
            test_images_root: The directory containing the test images. If ``None``, the directory containing the
                ``test_file`` will be used.
            predict_data_frame: The pandas ``DataFrame`` containing the data to use when predicting.
            predict_images_root: The directory containing the predict images. If ``None``, the directory containing the
                ``predict_file`` will be used.
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
            sampler: The ``sampler`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = ImageClassificationData.from_data_frame(
                "image_id",
                "target",
                train_data_frame=train_data,
                train_images_root="data/train_images",
            )
        """
        return cls.from_data_source(
            "data_frame",
            (train_data_frame, input_field, target_fields, train_images_root),
            (val_data_frame, input_field, target_fields, val_images_root),
            (test_data_frame, input_field, target_fields, test_images_root),
            (predict_data_frame, input_field, target_fields, predict_images_root),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[str] = None,
        train_images_root: Optional[str] = None,
        val_file: Optional[str] = None,
        val_images_root: Optional[str] = None,
        test_file: Optional[str] = None,
        test_images_root: Optional[str] = None,
        predict_file: Optional[str] = None,
        predict_images_root: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        **preprocess_kwargs: Any,
    ) -> 'DataModule':
        """Creates a :class:`~flash.image.classification.data.ImageClassificationData` object from the given CSV files
        using the :class:`~flash.core.data.data_source.DataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.CSV`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            input_field: The field (column) in the CSV file to use for the input.
            target_fields: The field or fields (columns) in the CSV file to use for the target.
            train_file: The CSV file containing the training data.
            train_images_root: The directory containing the train images. If ``None``, the directory containing the
                ``train_file`` will be used.
            val_file: The CSV file containing the validation data.
            val_images_root: The directory containing the validation images. If ``None``, the directory containing the
                ``val_file`` will be used.
            test_file: The CSV file containing the testing data.
            test_images_root: The directory containing the test images. If ``None``, the directory containing the
                ``test_file`` will be used.
            predict_file: The CSV file containing the data to use when predicting.
            predict_images_root: The directory containing the predict images. If ``None``, the directory containing the
                ``predict_file`` will be used.
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
            sampler: The ``sampler`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = ImageClassificationData.from_csv(
                "image_id",
                "target",
                train_file="train_data.csv",
                train_images_root="data/train_images",
            )
        """
        return cls.from_data_source(
            DefaultDataSources.CSV,
            (train_file, input_field, target_fields, train_images_root),
            (val_file, input_field, target_fields, val_images_root),
            (test_file, input_field, target_fields, test_images_root),
            (predict_file, input_field, target_fields, predict_images_root),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
        )

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return MatplotlibVisualization(*args, **kwargs)

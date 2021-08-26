from typing import Any, Callable, Dict, Optional, Type

from torch.utils.data import Sampler

from flash.core.data.base_viz import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import BaseDataFormat, DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Deserializer, Preprocess
from flash.pointcloud.detection.open3d_ml.data_sources import (
    PointCloudObjectDetectionDataFormat,
    PointCloudObjectDetectorFoldersDataSource,
)


class PointCloudObjectDetectorDatasetDataSource(DataSource):
    def __init__(self, **kwargs):
        super().__init__()

    def load_data(
        self,
        data: Any,
        dataset: Optional[Any] = None,
    ) -> Any:

        dataset.dataset = data

        return range(len(data))

    def load_sample(self, index: int, dataset: Optional[Any] = None) -> Any:
        sample = dataset.dataset[index]

        return {
            DefaultDataKeys.INPUT: sample["data"],
            DefaultDataKeys.METADATA: sample["attr"],
        }


class PointCloudObjectDetectorPreprocess(Preprocess):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        deserializer: Optional[Deserializer] = None,
        **data_source_kwargs,
    ):

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.DATASETS: PointCloudObjectDetectorDatasetDataSource(**data_source_kwargs),
                DefaultDataSources.FOLDERS: PointCloudObjectDetectorFoldersDataSource(**data_source_kwargs),
            },
            deserializer=deserializer,
            default_data_source=DefaultDataSources.FOLDERS,
        )

    def get_state_dict(self):
        return {}

    def state_dict(self):
        return {}

    @classmethod
    def load_state_dict(cls, state_dict, strict: bool = False):
        pass


class PointCloudObjectDetectorData(DataModule):

    preprocess_cls = PointCloudObjectDetectorPreprocess

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
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Type[Sampler]] = None,
        scans_folder_name: Optional[str] = "scans",
        labels_folder_name: Optional[str] = "labels",
        calibrations_folder_name: Optional[str] = "calibs",
        data_format: Optional[BaseDataFormat] = PointCloudObjectDetectionDataFormat.KITTI,
        **preprocess_kwargs: Any,
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
            scans_folder_name: The name of the pointcloud scan folder
            labels_folder_name: The name of the pointcloud scan labels folder
            calibrations_folder_name: The name of the pointcloud scan calibration folder
            data_format: Format in which the data are stored.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_folders(
                train_folder="train_folder",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.FOLDERS,
            train_folder,
            val_folder,
            test_folder,
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
            sampler=sampler,
            scans_folder_name=scans_folder_name,
            labels_folder_name=labels_folder_name,
            calibrations_folder_name=calibrations_folder_name,
            data_format=data_format,
            **preprocess_kwargs,
        )

from typing import Any, Callable, Dict, Optional, Tuple

from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Deserializer, Preprocess
from flash.core.utilities.imports import requires
from flash.pointcloud.segmentation.open3d_ml.sequences_dataset import SequencesDataset


class PointCloudSegmentationDatasetDataSource(DataSource):
    def load_data(
        self,
        data: Any,
        dataset: Optional[Any] = None,
    ) -> Any:
        if self.training:
            dataset.num_classes = len(data.dataset.label_to_names)

        dataset.dataset = data

        return range(len(data))

    def load_sample(self, index: int, dataset: Optional[Any] = None) -> Any:
        sample = dataset.dataset[index]

        return {
            DefaultDataKeys.INPUT: sample["data"],
            DefaultDataKeys.METADATA: sample["attr"],
        }


class PointCloudSegmentationFoldersDataSource(DataSource):
    @requires("pointcloud")
    def load_data(
        self,
        folder: Any,
        dataset: Optional[Any] = None,
    ) -> Any:
        sequence_dataset = SequencesDataset(folder, use_cache=True, predicting=self.predicting)
        dataset.dataset = sequence_dataset
        if self.training:
            dataset.num_classes = sequence_dataset.num_classes

        return range(len(sequence_dataset))

    def load_sample(self, index: int, dataset: Optional[Any] = None) -> Any:
        sample = dataset.dataset[index]

        return {
            DefaultDataKeys.INPUT: sample["data"],
            DefaultDataKeys.METADATA: sample["attr"],
        }


class PointCloudSegmentationPreprocess(Preprocess):
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
            data_sources={
                DefaultDataSources.DATASETS: PointCloudSegmentationDatasetDataSource(),
                DefaultDataSources.FOLDERS: PointCloudSegmentationFoldersDataSource(),
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


class PointCloudSegmentationData(DataModule):

    preprocess_cls = PointCloudSegmentationPreprocess

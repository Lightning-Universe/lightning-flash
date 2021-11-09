from typing import Any, Callable, Dict, Optional, Tuple

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input, InputDataKeys, InputFormat
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.process import Deserializer
from flash.core.utilities.imports import requires
from flash.pointcloud.segmentation.open3d_ml.sequences_dataset import SequencesDataset


class PointCloudSegmentationDatasetInput(Input):
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
            InputDataKeys.INPUT: sample["data"],
            InputDataKeys.METADATA: sample["attr"],
        }


class PointCloudSegmentationFoldersInput(Input):
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
            InputDataKeys.INPUT: sample["data"],
            InputDataKeys.METADATA: sample["attr"],
        }


class PointCloudSegmentationInputTransform(InputTransform):
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
                InputFormat.DATASETS: PointCloudSegmentationDatasetInput(),
                InputFormat.FOLDERS: PointCloudSegmentationFoldersInput(),
            },
            deserializer=deserializer,
            default_=InputFormat.FOLDERS,
        )

    def get_state_dict(self):
        return {}

    def state_dict(self):
        return {}

    @classmethod
    def load_state_dict(cls, state_dict, strict: bool = False):
        pass


class PointCloudSegmentationData(DataModule):

    input_transform_cls = PointCloudSegmentationInputTransform

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import Deserializer
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess


class PointCloudClassificationDatasetDataSource(DataSource):

    def load_data(
        self,
        data: Any,
        dataset: Optional[Any] = None,
    ) -> Any:
        if self.training:
            dataset.num_classes = len(data.dataset.label_to_names)

        dataset.dataset = data

        return range(len(data))

    @staticmethod
    def load_sample(sample: Mapping[str, Any], dataset: Optional[Any] = None) -> Any:
        return {
            DefaultDataKeys.INPUT: dataset.dataset.get_data(sample)["point"],
            DefaultDataKeys.TARGET: dataset.dataset.get_data(sample)["label"]
        }


class PointCloudClassificationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        deserializer: Optional[Deserializer] = None,
        **data_source_kwargs: Any,
    ):
        self.image_size = image_size

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.DATASET: PointCloudClassificationDatasetDataSource(**data_source_kwargs),
            },
            deserializer=deserializer,
            default_data_source=DefaultDataSources.DATASET,
        )

    def get_state_dict(self):
        return {}

    def state_dict(self):
        return {}

    @classmethod
    def load_state_dict(cls, state_dict, strict: bool):
        pass


class PointCloudClassificationData(DataModule):

    preprocess_cls = PointCloudClassificationPreprocess

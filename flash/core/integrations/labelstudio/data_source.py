import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TypeVar, Union

import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.cloud_io import get_filesystem

from flash import DataSource
from flash.core.data.auto_dataset import AutoDataset, IterableAutoDataset
from flash.core.data.data_source import DefaultDataKeys, has_len
from flash.core.utilities.imports import _PYTORCHVIDEO_AVAILABLE, _TEXT_AVAILABLE, _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets.folder import default_loader
DATA_TYPE = TypeVar("DATA_TYPE")


class LabelStudioDataSource(DataSource):
    """The ``LabelStudioDatasource`` expects the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be a json export from label studio."""

    def __init__(self):
        super().__init__()
        self.results = []
        self.test_results = []
        self.val_results = []
        self.classes = set()
        self.data_types = set()
        self.tag_types = set()
        self.num_classes = 0
        self._data_folder = ""
        self._raw_data = {}
        self.multi_label = False
        self.split = None

    def load_data(self, data: Optional[Any] = None, dataset: Optional[Any] = None) -> Sequence[Mapping[str, Any]]:
        """Iterate through all tasks in exported data and construct train\test\val results."""
        if data and isinstance(data, dict):
            data_folder = data.get("data_folder")
            file_path = data.get("export_json")
            fs = get_filesystem(file_path)
            with fs.open(file_path) as f:
                _raw_data = json.load(f)
            self.multi_label = data.get("multi_label", False)
            self.split = data.get("split")
            results, test_results, classes, data_types, tag_types = LabelStudioDataSource._load_json_data(
                _raw_data, data_folder=data_folder, multi_label=self.multi_label
            )
            self.classes = self.classes | classes
            self.data_types = self.data_types | data_types
            self.num_classes = len(self.classes)
            self.tag_types = self.tag_types | tag_types
            # splitting result to train and val sets
            if self.split:
                import random

                random.shuffle(results)
                prop = int(len(results) * self.split)
                self.val_results = results[:prop]
                self.results = results[prop:]
                self.test_results = test_results
                return self.results
            return results + test_results
        return []

    def load_sample(self, sample: Mapping[str, Any] = None, dataset: Optional[Any] = None) -> Any:
        """Load 1 sample from dataset."""
        # all other data types
        # separate label from data
        label = self._get_labels_from_sample(sample["label"])
        # delete label from input data
        del sample["label"]
        result = {
            DefaultDataKeys.INPUT: sample,
            DefaultDataKeys.TARGET: label,
        }
        return result

    def generate_dataset(
        self,
        data: Optional[DATA_TYPE],
        running_stage: RunningStage,
    ) -> Optional[Union[AutoDataset, IterableAutoDataset]]:
        """Generate dataset from loaded data."""
        res = self.load_data(data)
        if running_stage in (RunningStage.TRAINING, RunningStage.TUNING):
            dataset = res
        elif running_stage == RunningStage.TESTING:
            dataset = res or self.test_results
        elif running_stage == RunningStage.PREDICTING:
            dataset = res or []
        elif running_stage == RunningStage.VALIDATING:
            dataset = res or self.val_results

        if has_len(dataset):
            dataset = AutoDataset(dataset, self, running_stage)
        else:
            dataset = IterableAutoDataset(dataset, self, running_stage)
        dataset.num_classes = self.num_classes
        return dataset

    def _get_labels_from_sample(self, labels):
        """Translate string labels to int."""
        sorted_labels = sorted(list(self.classes))
        if isinstance(labels, list):
            label = []
            for item in labels:
                label.append(sorted_labels.index(item))
        else:
            label = sorted_labels.index(labels)
        return label

    @staticmethod
    def _load_json_data(data, data_folder, multi_label=False):
        """Utility method to extract data from Label Studio json files."""
        results = []
        test_results = []
        data_types = set()
        tag_types = set()
        classes = set()
        for task in data:
            for annotation in task["annotations"]:
                # extracting data types from tasks
                for key in task.get("data"):
                    data_types.add(key)
                # Adding ground_truth annotation to separate dataset
                result = annotation["result"]
                for res in result:
                    t = res["type"]
                    tag_types.add(t)
                    for label in res["value"][t]:
                        # check if labeling result is a list of labels
                        if isinstance(label, list) and not multi_label:
                            for sublabel in label:
                                classes.add(sublabel)
                                temp = {}
                                temp["file_upload"] = task.get("file_upload")
                                temp["data"] = task.get("data")
                                if temp["file_upload"]:
                                    temp["file_upload"] = os.path.join(data_folder, temp["file_upload"])
                                else:
                                    for key in temp["data"]:
                                        p = temp["data"].get(key)
                                    path = Path(p)
                                    if path and data_folder:
                                        temp["file_upload"] = os.path.join(data_folder, path.name)
                                temp["label"] = sublabel
                                temp["result"] = res.get("value")
                                if annotation["ground_truth"]:
                                    test_results.append(temp)
                                elif not annotation["ground_truth"]:
                                    results.append(temp)
                        else:
                            if isinstance(label, list):
                                for item in label:
                                    classes.add(item)
                            else:
                                classes.add(label)
                            temp = {}
                            temp["file_upload"] = task.get("file_upload")
                            temp["data"] = task.get("data")
                            if temp["file_upload"] and data_folder:
                                temp["file_upload"] = os.path.join(data_folder, temp["file_upload"])
                            else:
                                for key in temp["data"]:
                                    p = temp["data"].get(key)
                                path = Path(p)
                                if path and data_folder:
                                    temp["file_upload"] = os.path.join(data_folder, path.name)
                            temp["label"] = label
                            temp["result"] = res.get("value")
                            if annotation["ground_truth"]:
                                test_results.append(temp)
                            elif not annotation["ground_truth"]:
                                results.append(temp)
        return results, test_results, classes, data_types, tag_types


class LabelStudioImageClassificationDataSource(LabelStudioDataSource):
    """The ``LabelStudioImageDataSource`` expects the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be a json export from label studio.
    Export data should point to image files"""

    def load_sample(self, sample: Mapping[str, Any] = None, dataset: Optional[Any] = None) -> Any:
        """Load 1 sample from dataset."""
        p = sample["file_upload"]
        # loading image
        image = default_loader(p)
        result = {DefaultDataKeys.INPUT: image, DefaultDataKeys.TARGET: self._get_labels_from_sample(sample["label"])}
        return result


class LabelStudioTextClassificationDataSource(LabelStudioDataSource):
    """The ``LabelStudioTextDataSource`` expects the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be a json export from label studio.
    Export data should point to text data
    """

    def __init__(self, backbone=None, max_length=128):
        super().__init__()
        if backbone:
            if _TEXT_AVAILABLE:
                from transformers import AutoTokenizer
            self.backbone = backbone
            self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
            self.max_length = max_length

    def load_sample(self, sample: Mapping[str, Any] = None, dataset: Optional[Any] = None) -> Any:
        """Load 1 sample from dataset."""
        if self.backbone:
            data = ""
            for key in sample.get("data"):
                data += sample.get("data").get(key)
            tokenized_data = self.tokenizer(data, max_length=self.max_length, truncation=True, padding="max_length")
            for key in tokenized_data:
                tokenized_data[key] = torch.tensor(tokenized_data[key])
            tokenized_data["labels"] = self._get_labels_from_sample(sample["label"])
            # separate text data type block
            result = tokenized_data
        return result


class LabelStudioVideoClassificationDataSource(LabelStudioDataSource):
    """The ``LabelStudioVideoDataSource`` expects the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be a json export from label studio.
    Export data should point to video files"""

    def __init__(self, video_sampler=None, clip_sampler=None, decode_audio=False, decoder: str = "pyav"):
        if not _PYTORCHVIDEO_AVAILABLE:
            raise ModuleNotFoundError("Please, run `pip install pytorchvideo`.")
        super().__init__()
        self.video_sampler = video_sampler or torch.utils.data.RandomSampler
        self.clip_sampler = clip_sampler
        self.decode_audio = decode_audio
        self.decoder = decoder

    def load_sample(self, sample: Mapping[str, Any] = None, dataset: Optional[Any] = None) -> Any:
        """Load 1 sample from dataset."""
        return sample

    def load_data(self, data: Optional[Any] = None, dataset: Optional[Any] = None) -> Sequence[Mapping[str, Any]]:
        """load_data produces a sequence or iterable of samples."""
        res = super().load_data(data, dataset)
        return self.convert_to_encodedvideo(res)

    def convert_to_encodedvideo(self, dataset):
        """Converting dataset to EncodedVideoDataset."""
        if len(dataset) > 0:
            from pytorchvideo.data import LabeledVideoDataset

            dataset = LabeledVideoDataset(
                [
                    (
                        os.path.join(self._data_folder, sample["file_upload"]),
                        {"label": self._get_labels_from_sample(sample["label"])},
                    )
                    for sample in dataset
                ],
                self.clip_sampler,
                decode_audio=self.decode_audio,
                decoder=self.decoder,
            )
            return dataset
        return []

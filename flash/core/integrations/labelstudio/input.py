import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Type

import numpy as np
import torch
from pytorch_lightning.utilities.cloud_io import get_filesystem
from torch.utils.data import Sampler

from flash.core.data.io.input import DataKeys, Input, IterableInput
from flash.core.data.properties import Properties
from flash.core.data.utilities.loading import load_image
from flash.core.utilities.imports import _PYTORCHVIDEO_AVAILABLE
from flash.core.utilities.stages import RunningStage

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.data.clip_sampling import make_clip_sampler


@dataclass(unsafe_hash=True, frozen=True)
class LabelStudioParameters:
    """The ``LabelStudioParameters`` stores the metadata loaded from the data."""

    multi_label: bool
    num_classes: Optional[int]
    classes: Set
    data_types: Set
    tag_types: Set


def _get_labels_from_sample(labels, classes):
    """Translate string labels to int."""
    sorted_labels = sorted(list(classes))
    return [sorted_labels.index(item) for item in labels] if isinstance(labels, list) else sorted_labels.index(labels)


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
                            temp = {
                                "file_upload": task.get("file_upload"),
                                "data": task.get("data"),
                                "label": sublabel,
                                "result": res.get("value"),
                            }
                            if temp["file_upload"]:
                                temp["file_upload"] = os.path.join(data_folder, temp["file_upload"])
                            else:
                                for key in temp["data"]:
                                    p = temp["data"].get(key)
                                path = Path(p)
                                if path and data_folder:
                                    temp["file_upload"] = os.path.join(data_folder, path.name)
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
                        temp = {
                            "file_upload": task.get("file_upload"),
                            "data": task.get("data"),
                            "label": label,
                            "result": res.get("value"),
                        }
                        if temp["file_upload"] and data_folder:
                            temp["file_upload"] = os.path.join(data_folder, temp["file_upload"])
                        else:
                            for key in temp["data"]:
                                p = temp["data"].get(key)
                            path = Path(p)
                            if path and data_folder:
                                temp["file_upload"] = os.path.join(data_folder, path.name)
                        if annotation["ground_truth"]:
                            test_results.append(temp)
                        elif not annotation["ground_truth"]:
                            results.append(temp)
    return results, test_results, classes, data_types, tag_types


class BaseLabelStudioInput(Properties):
    parameters: LabelStudioParameters

    def load_data(
        self, data: Optional[Any], parameters: Optional[LabelStudioParameters] = None
    ) -> Sequence[Mapping[str, Any]]:
        """Iterate through all tasks in exported data and construct train\test\val results."""
        if data and isinstance(data, dict):
            data_folder = data.get("data_folder")
            file_path = data.get("export_json")
            multi_label = data.get("multi_label", False)
            fs = get_filesystem(file_path)
            with fs.open(file_path) as f:
                _raw_data = json.load(f)
            results, test_results, classes, data_types, tag_types = _load_json_data(
                _raw_data, data_folder=data_folder, multi_label=multi_label
            )
            if self.training:
                self.parameters = LabelStudioParameters(
                    classes=classes,
                    data_types=data_types,
                    tag_types=tag_types,
                    multi_label=multi_label,
                    num_classes=len(classes),
                )
            else:
                self.parameters = parameters
            return test_results if self.testing else results
        return []

    def load_sample(self, sample: Mapping[str, Any] = None) -> Any:
        """Load 1 sample from dataset."""
        # all other data types
        # separate label from data
        label = _get_labels_from_sample(sample["label"], self.parameters.classes)
        # delete label from input data
        del sample["label"]
        result = {
            DataKeys.INPUT: sample,
            DataKeys.TARGET: label,
        }
        return result

    @staticmethod
    def _split_train_test_data(data: Dict, multi_label: bool = False) -> List[Dict]:
        file_path = data.get("export_json", None)

        if not file_path:
            raise TypeError("The key `export_json` should be provided as a string.")

        fs = get_filesystem(file_path)
        with fs.open(file_path) as f:
            raw_data = np.asarray(json.load(f))

        train_raw_data = []
        test_raw_data = []
        for task in raw_data:
            for annotation in task["annotations"]:
                if annotation["ground_truth"]:
                    test_raw_data.append(task)
                elif not annotation["ground_truth"]:
                    train_raw_data.append(task)
                break

        assert len(raw_data) == len(train_raw_data) + len(test_raw_data)

        dirname = os.path.dirname(file_path)
        basename = os.path.basename(file_path)
        results = []
        for stage, raw_data in [("train", train_raw_data), ("test", test_raw_data)]:
            filename = basename if stage in basename else f"{stage}_{basename}"
            export_path = os.path.join(dirname, filename)
            LabelStudioInput._export_data_to_json(export_path, raw_data)
            output_data = deepcopy(data)
            output_data["export_json"] = export_path
            results.append(output_data)
        return results

    @staticmethod
    def _export_data_to_json(export_path: str, raw_data: List[Dict]) -> Dict:
        fs = get_filesystem(export_path)
        if fs.exists(export_path):
            fs.delete(export_path)
        with fs.open(export_path, mode="w") as f:
            json.dump(raw_data, f)

    @staticmethod
    def _split_train_val_data(data: Dict, split: float = 0) -> List[Dict]:
        assert split > 0 and split < 1
        file_path = data.get("export_json", None)

        if not file_path:
            raise TypeError("The key `export_json` should be provided as a string.")

        fs = get_filesystem(file_path)
        with fs.open(file_path) as f:
            raw_data = np.asarray(json.load(f))

        L = len(raw_data)
        indices = np.random.permutation(L)
        limit = int(L * split)
        train_raw_data = raw_data[indices[limit:]]
        val_raw_data = raw_data[indices[:limit]]

        dirname = os.path.dirname(file_path)
        basename = os.path.basename(file_path)
        results = []
        for stage, raw_data in [("train", train_raw_data), ("val", val_raw_data)]:
            filename = basename if stage in basename else f"{stage}_{basename}"
            export_path = os.path.join(dirname, filename)
            LabelStudioInput._export_data_to_json(export_path, raw_data.tolist())
            output_data = deepcopy(data)
            output_data["export_json"] = export_path
            results.append(output_data)
        return results


class LabelStudioInput(BaseLabelStudioInput, Input):
    """The ``LabelStudioInput`` expects the input to
    :meth:`~flash.core.data.io.input.Input.load_data` to be a json export from label studio."""


class LabelStudioIterableInput(BaseLabelStudioInput, IterableInput):
    """The ``LabelStudioInput`` expects the input to
    :meth:`~flash.core.data.io.input.Input.load_data` to be a json export from label studio."""


class LabelStudioImageClassificationInput(LabelStudioInput):
    """The ``LabelStudioImageInput`` expects the input to
    :meth:`~flash.core.data.io.input.Input.load_data` to be a json export from label studio.
    Export data should point to image files"""

    def load_sample(self, sample: Mapping[str, Any] = None) -> Any:
        """Load 1 sample from dataset."""
        p = sample["file_upload"]
        # loading image
        image = load_image(p)
        result = {
            DataKeys.INPUT: image,
            DataKeys.TARGET: _get_labels_from_sample(sample["label"], self.parameters.classes),
        }
        return result


class LabelStudioTextClassificationInput(LabelStudioInput):
    """The ``LabelStudioTextInput`` expects the input to
    :meth:`~flash.core.data.io.input.Input.load_data` to be a json export from label studio.
    Export data should point to text data
    """

    def load_sample(self, sample: Mapping[str, Any] = None) -> Any:
        """Load 1 sample from dataset."""
        data = ""
        for key in sample.get("data"):
            data += sample.get("data").get(key)
        return {
            DataKeys.INPUT: data,
            DataKeys.TARGET: _get_labels_from_sample(sample["label"], self.parameters.classes),
        }


class LabelStudioVideoClassificationInput(LabelStudioIterableInput):
    """The ``LabelStudioVideoInput`` expects the input to
    :meth:`~flash.core.data.io.input.Input.load_data` to be a json export from label studio.
    Export data should point to video files"""

    def __init__(
        self,
        running_stage: RunningStage,
        data: Any,
        *args,
        clip_sampler: str = "random",
        clip_duration: float = 2,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio=False,
        decoder: str = "pyav",
        clip_sampler_kwargs: Optional[Dict] = None,
        data_folder: str = "",
        **kwargs,
    ):
        if not _PYTORCHVIDEO_AVAILABLE:
            raise ModuleNotFoundError("Please, run `pip install pytorchvideo`.")
        self.video_sampler = video_sampler or torch.utils.data.RandomSampler
        clip_sampler_kwargs = clip_sampler_kwargs or {}
        self.clip_sampler = make_clip_sampler(clip_sampler, clip_duration, **clip_sampler_kwargs)
        self.decode_audio = decode_audio
        self.decoder = decoder
        self.clip_duration = clip_duration
        self._data_folder = data_folder
        super().__init__(running_stage, data, *args, **kwargs)

    def load_sample(self, sample: Mapping[str, Any] = None) -> Any:
        """Load 1 sample from dataset."""
        return sample

    def load_data(
        self, data: Optional[Any] = None, parameters: Optional[LabelStudioParameters] = None
    ) -> Sequence[Mapping[str, Any]]:
        """load_data produces a sequence or iterable of samples."""
        res = super().load_data(data, parameters=parameters)
        return self.convert_to_encodedvideo(res)

    def convert_to_encodedvideo(self, dataset):
        """Converting dataset to EncodedVideoDataset."""
        if len(dataset) > 0:

            from pytorchvideo.data import LabeledVideoDataset

            dataset = LabeledVideoDataset(
                [
                    (
                        os.path.join(self._data_folder, sample["file_upload"]),
                        {"label": _get_labels_from_sample(sample["label"], self.parameters.classes)},
                    )
                    for sample in dataset
                ],
                clip_sampler=self.clip_sampler,
                decode_audio=self.decode_audio,
                decoder=self.decoder,
            )
            return dataset
        return []


def _parse_labelstudio_arguments(
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
    val_split: Optional[float] = None,
    multi_label: Optional[bool] = False,
):

    train_data = None
    val_data = None
    test_data = None
    predict_data = None
    data = {
        "data_folder": data_folder,
        "export_json": export_json,
        "multi_label": multi_label,
    }

    if (train_data_folder or data_folder) and train_export_json:
        train_data = {
            "data_folder": train_data_folder or data_folder,
            "export_json": train_export_json,
            "multi_label": multi_label,
        }
    if (val_data_folder or data_folder) and val_export_json:
        val_data = {
            "data_folder": val_data_folder or data_folder,
            "export_json": val_export_json,
            "multi_label": multi_label,
        }
    if (test_data_folder or data_folder) and test_export_json:
        test_data = {
            "data_folder": test_data_folder or data_folder,
            "export_json": test_export_json,
            "multi_label": multi_label,
        }
    if (predict_data_folder or data_folder) and predict_export_json:
        predict_data = {
            "data_folder": predict_data_folder or data_folder,
            "export_json": predict_export_json,
            "multi_label": multi_label,
        }

    train_data = train_data if train_data else data

    # TODO: Extract test from data if present.

    if val_split and val_data is None:
        train_data, val_data = LabelStudioInput._split_train_val_data(train_data, val_split)

    return train_data, val_data, test_data, predict_data

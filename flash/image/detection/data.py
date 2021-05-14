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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.utilities.imports import _COCO_AVAILABLE, _TORCHVISION_AVAILABLE
from flash.image.data import ImagePathsDataSource
from flash.image.detection.transforms import default_transforms

if _COCO_AVAILABLE:
    from pycocotools.coco import COCO

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets.folder import default_loader


class COCODataSource(DataSource[Tuple[str, str]]):

    def load_data(self, data: Tuple[str, str], dataset: Optional[Any] = None) -> Sequence[Dict[str, Any]]:
        root, ann_file = data

        coco = COCO(ann_file)

        categories = coco.loadCats(coco.getCatIds())
        if categories:
            dataset.num_classes = categories[-1]["id"] + 1

        img_ids = list(sorted(coco.imgs.keys()))
        paths = coco.loadImgs(img_ids)

        data = []

        for img_id, path in zip(img_ids, paths):
            path = path["file_name"]

            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)

            boxes, labels, areas, iscrowd = [], [], [], []

            # Ref: https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
            if self.training and all(any(o <= 1 for o in obj["bbox"][2:]) for obj in annotations):
                continue

            for obj in annotations:
                xmin = obj["bbox"][0]
                ymin = obj["bbox"][1]
                xmax = xmin + obj["bbox"][2]
                ymax = ymin + obj["bbox"][3]

                bbox = [xmin, ymin, xmax, ymax]
                keep = (bbox[3] > bbox[1]) & (bbox[2] > bbox[0])
                if keep:
                    boxes.append(bbox)
                    labels.append(obj["category_id"])
                    areas.append(obj["area"])
                    iscrowd.append(obj["iscrowd"])

            data.append(
                dict(
                    input=os.path.join(root, path),
                    target=dict(
                        boxes=boxes,
                        labels=labels,
                        image_id=img_id,
                        area=areas,
                        iscrowd=iscrowd,
                    )
                )
            )
        return data

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DefaultDataKeys.INPUT] = default_loader(sample[DefaultDataKeys.INPUT])
        return sample


class ObjectDetectionPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.FILES: ImagePathsDataSource(),
                DefaultDataSources.FOLDERS: ImagePathsDataSource(),
                "coco": COCODataSource(),
            },
            default_data_source=DefaultDataSources.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms()


class ObjectDetectionData(DataModule):

    preprocess_cls = ObjectDetectionPreprocess

    @classmethod
    def from_coco(
        cls,
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_ann_file: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **preprocess_kwargs: Any,
    ):
        """Creates a :class:`~flash.image.detection.data.ObjectDetectionData` object from the given data
        folders and corresponding target folders.

        Args:
            train_folder: The folder containing the train data.
            train_ann_file: The COCO format annotation file.
            val_folder: The folder containing the validation data.
            val_ann_file: The COCO format annotation file.
            test_folder: The folder containing the test data.
            test_ann_file: The COCO format annotation file.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = SemanticSegmentationData.from_coco(
                train_folder="train_folder",
                train_ann_file="annotations.json",
            )
        """
        return cls.from_data_source(
            "coco",
            (train_folder, train_ann_file) if train_folder else None,
            (val_folder, val_ann_file) if val_folder else None,
            (test_folder, test_ann_file) if test_folder else None,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            **preprocess_kwargs,
        )

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
from typing import Any, Callable, List, Optional, Tuple

import torch
from PIL import Image
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from torch._six import container_abcs
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms as T

from flash.core.data import TaskDataPipeline
from flash.core.data.datamodule import DataModule
from flash.core.data.utils import _contains_any_tensor
from flash.utils.imports import _COCO_AVAILABLE
from flash.vision.classification.data import _pil_loader

if _COCO_AVAILABLE:
    from pycocotools.coco import COCO


class CustomCOCODataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
    ):
        if not _COCO_AVAILABLE:
            raise ImportError("Kindly install the COCO API `pycocotools` to use the Dataset")

        self.root = root
        self.transforms = transforms
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    @property
    def num_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        if not categories:
            raise ValueError("No Categories found")
        return categories[-1]["id"] + 1

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        coco = self.coco
        img_idx = self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_idx)
        annotations = coco.loadAnns(ann_ids)

        image_path = coco.loadImgs(img_idx)[0]["file_name"]
        img = Image.open(os.path.join(self.root, image_path))

        boxes = []
        labels = []
        areas = []
        iscrowd = []

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

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([img_idx])
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def _coco_remove_images_without_annotations(dataset):
    # Ref: https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py

    def _has_only_empty_bbox(annot: List):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in annot)

    def _has_valid_annotation(annot: List):
        # if it's empty, there is no annotation
        if not annot:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(annot):
            return False
        return True

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


_default_transform = T.ToTensor()


class ObjectDetectionDataPipeline(TaskDataPipeline):

    def __init__(self, valid_transform: Optional[Callable] = _default_transform, loader: Callable = _pil_loader):
        self._valid_transform = valid_transform
        self._loader = loader

    def before_collate(self, samples: Any) -> Any:
        if _contains_any_tensor(samples):
            return samples

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                output = self._loader(sample)
                outputs.append(self._valid_transform(output))
            return outputs
        raise MisconfigurationException("The samples should either be a tensor, a list of paths or a path.")

    def collate(self, samples: Any) -> Any:
        if not isinstance(samples, Tensor):
            elem = samples[0]
            if isinstance(elem, container_abcs.Sequence):
                return tuple(zip(*samples))
            return default_collate(samples)
        return samples.unsqueeze(dim=0)


class ObjectDetectionData(DataModule):

    @classmethod
    def from_coco(
        cls,
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        train_transform: Optional[Callable] = _default_transform,
        valid_folder: Optional[str] = None,
        valid_ann_file: Optional[str] = None,
        valid_transform: Optional[Callable] = _default_transform,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        test_transform: Optional[Callable] = _default_transform,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **kwargs
    ):
        train_ds = CustomCOCODataset(train_folder, train_ann_file, train_transform)
        num_classes = train_ds.num_classes
        train_ds = _coco_remove_images_without_annotations(train_ds)

        valid_ds = (
            CustomCOCODataset(valid_folder, valid_ann_file, valid_transform) if valid_folder is not None else None
        )

        test_ds = (CustomCOCODataset(test_folder, test_ann_file, test_transform) if test_folder is not None else None)

        datamodule = cls(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        datamodule.num_classes = num_classes
        datamodule.data_pipeline = ObjectDetectionDataPipeline()
        return datamodule

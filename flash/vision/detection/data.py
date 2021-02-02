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
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms as T

from flash.core.data.datamodule import DataModule


class CustomCOCODataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
    ):
        from pycocotools.coco import COCO

        self.root = root
        self.transforms = transforms
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    @property
    def num_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        return len(categories)

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

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj["category_id"] - 1)
            areas.append(obj["area"])
            iscrowd.append(obj["iscrowd"])

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([img_idx])
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        target["iscrowd"] = iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


_default_transform = T.Compose([T.ToTensor()])


class ImageDetectionData(DataModule):

    @classmethod
    def coco_format(
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

        datamodule.num_classes = train_ds.num_classes
        return datamodule

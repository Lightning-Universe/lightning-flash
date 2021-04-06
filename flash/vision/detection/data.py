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
from typing import Any, Callable, List, Optional, Tuple, Type

import torch
from PIL import Image
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, tensor
from torch._six import container_abcs
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms as T

from flash.data.auto_dataset import AutoDataset
from flash.data.data_module import DataModule
from flash.data.process import Preprocess
from flash.data.utils import _contains_any_tensor
from flash.utils.imports import _COCO_AVAILABLE
from flash.vision.utils import pil_loader

if _COCO_AVAILABLE:
    from pycocotools.coco import COCO


class CustomCOCODataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        loader: Optional[Callable] = pil_loader,
    ):
        if not _COCO_AVAILABLE:
            raise ImportError("Kindly install the COCO API `pycocotools` to use the Dataset")

        self.root = root
        self.transforms = transforms
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.loader = loader

    @property
    def num_classes(self) -> int:
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

        target = dict(
            boxes=torch.as_tensor(boxes, dtype=torch.float32),
            labels=torch.as_tensor(labels, dtype=torch.int64),
            image_id=tensor([img_idx]),
            area=torch.as_tensor(areas, dtype=torch.float32),
            iscrowd=torch.as_tensor(iscrowd, dtype=torch.int64)
        )

        if self.transforms:
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


class ObjectDetectionPreprocess(Preprocess):

    to_tensor = T.ToTensor()

    def load_data(self, metadata: Any, dataset: AutoDataset) -> CustomCOCODataset:
        # Extract folder, coco annotation file and the transform to be applied on the images
        folder, ann_file, transform = metadata
        ds = CustomCOCODataset(folder, ann_file, transform)
        if self.training:
            dataset.num_classes = ds.num_classes
            ds = _coco_remove_images_without_annotations(ds)
        return ds

    def predict_load_data(self, samples):
        return samples

    def pre_tensor_transform(self, samples: Any) -> Any:
        if _contains_any_tensor(samples):
            return samples

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                outputs.append(pil_loader(sample))
            return outputs
        raise MisconfigurationException("The samples should either be a tensor, a list of paths or a path.")

    def predict_to_tensor_transform(self, sample) -> Any:
        return self.to_tensor(sample[0])

    def collate(self, samples: Any) -> Any:
        if not isinstance(samples, Tensor):
            elem = samples[0]
            if isinstance(elem, container_abcs.Sequence):
                return tuple(zip(*samples))
            return default_collate(samples)
        return samples.unsqueeze(dim=0)


class ObjectDetectionData(DataModule):

    preprocess_cls = ObjectDetectionPreprocess

    @classmethod
    def instantiate_preprocess(
        cls,
        train_transform: Optional[Callable],
        val_transform: Optional[Callable],
        preprocess_cls: Type[Preprocess] = None
    ) -> Preprocess:

        preprocess_cls = preprocess_cls or cls.preprocess_cls
        return preprocess_cls(train_transform, val_transform)

    @classmethod
    def from_coco(
        cls,
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        train_transform: Optional[Callable] = _default_transform,
        val_folder: Optional[str] = None,
        val_ann_file: Optional[str] = None,
        val_transform: Optional[Callable] = _default_transform,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        test_transform: Optional[Callable] = _default_transform,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        preprocess_cls: Type[Preprocess] = None,
        **kwargs
    ):

        preprocess = cls.instantiate_preprocess(train_transform, val_transform, preprocess_cls=preprocess_cls)

        datamodule = cls.from_load_data_inputs(
            train_load_data_input=(train_folder, train_ann_file, train_transform),
            val_load_data_input=(val_folder, val_ann_file, val_transform) if val_folder else None,
            test_load_data_input=(test_folder, test_ann_file, test_transform) if test_folder else None,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            **kwargs
        )
        datamodule.num_classes = datamodule._train_ds.num_classes
        return datamodule

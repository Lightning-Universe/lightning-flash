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

from pytorch_lightning.trainer.states import RunningStage
from torch.nn import Module
from torchvision.datasets.folder import default_loader

from flash.data.data_module import DataModule
from flash.data.data_source import DataSource
from flash.data.process import Preprocess
from flash.utils.imports import _COCO_AVAILABLE
from flash.vision.detection.transforms import default_transforms

if _COCO_AVAILABLE:
    from pycocotools.coco import COCO


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
        sample['input'] = default_loader(sample['input'])
        return sample


class ObjectDetectionPreprocess(Preprocess):

    data_sources = {
        "coco": COCODataSource,
    }

    def collate(self, samples: Any) -> Any:
        return {key: [sample[key] for sample in samples] for key in samples[0]}

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "train_transform": self._train_transform,
            "val_transform": self._val_transform,
            "test_transform": self._test_transform,
            "predict_transform": self._predict_transform,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def default_train_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms()

    def default_val_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms()


class ObjectDetectionData(DataModule):

    preprocess_cls = ObjectDetectionPreprocess

    @classmethod
    def from_coco(
        cls,
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Module]] = None,
        val_folder: Optional[str] = None,
        val_ann_file: Optional[str] = None,
        val_transform: Optional[Dict[str, Module]] = None,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        test_transform: Optional[Dict[str, Module]] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        preprocess: Preprocess = None,
        **kwargs
    ):
        preprocess = preprocess or cls.preprocess_cls(
            train_transform,
            val_transform,
            test_transform,
        )

        data_source = preprocess.data_source_of_type(COCODataSource)()

        return cls.from_data_source(
            data_source=data_source,
            train_data=(train_folder, train_ann_file) if train_folder else None,
            val_data=(val_folder, val_ann_file) if val_folder else None,
            test_data=(test_folder, test_ann_file) if test_folder else None,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            **kwargs
        )

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
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys, InputFormat
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _FASTFACE_AVAILABLE, _TORCHVISION_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.image.classification.input import ImageClassificationFilesInput, ImageClassificationFolderInput
from flash.image.data import ImageInput

if _TORCHVISION_AVAILABLE:
    import torchvision

if _FASTFACE_AVAILABLE:
    import fastface as ff


def fastface_collate_fn(samples: Sequence[Dict[str, Any]]) -> Dict[str, Sequence[Any]]:
    """Collate function from fastface.

    Organizes individual elements in a batch, calls prepare_batch from fastface and prepares the targets.
    """
    samples = {key: [sample[key] for sample in samples] for key in samples[0]}

    images, scales, paddings = ff.utils.preprocess.prepare_batch(samples[DataKeys.INPUT], None, adaptive_batch=True)

    samples["scales"] = scales
    samples["paddings"] = paddings

    if DataKeys.TARGET in samples.keys():
        targets = samples[DataKeys.TARGET]
        targets = [{"target_boxes": target["boxes"]} for target in targets]

        for i, (target, scale, padding) in enumerate(zip(targets, scales, paddings)):
            target["target_boxes"] *= scale
            target["target_boxes"][:, [0, 2]] += padding[0]
            target["target_boxes"][:, [1, 3]] += padding[1]
            targets[i]["target_boxes"] = target["target_boxes"]

        samples[DataKeys.TARGET] = targets
    samples[DataKeys.INPUT] = images

    return samples


class FastFaceInput(ImageInput):
    """Logic for loading from FDDBDataset."""

    def load_data(self, dataset: Dataset) -> List[Dict[str, Any]]:
        return [
            {
                DataKeys.INPUT: filepath,
                "boxes": targets["target_boxes"],
                "labels": [1] * targets["target_boxes"].shape[0],
            }
            for filepath, targets in zip(dataset.ids, dataset.targets)
        ]


class FaceDetectionInputTransform(InputTransform):
    """Applies default transform and collate_fn for fastface on FastFaceDataSource."""

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
            inputs={
                InputFormat.FILES: ImageClassificationFilesInput,
                InputFormat.FOLDERS: ImageClassificationFolderInput,
                InputFormat.DATASETS: FastFaceInput,
            },
            default_input=InputFormat.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def default_transforms(self) -> Dict[str, Callable]:
        return {
            "per_sample_transform": nn.Sequential(
                ApplyToKeys(DataKeys.INPUT, torchvision.transforms.ToTensor()),
                ApplyToKeys(
                    DataKeys.TARGET,
                    nn.Sequential(
                        ApplyToKeys("boxes", torch.as_tensor),
                        ApplyToKeys("labels", torch.as_tensor),
                    ),
                ),
            ),
            "collate": fastface_collate_fn,
        }


class FaceDetectionOutputTransform(OutputTransform):
    """Generates preds from model output."""

    @staticmethod
    def per_batch_transform(batch: Any) -> Any:
        scales = batch["scales"]
        paddings = batch["paddings"]

        batch.pop("scales", None)
        batch.pop("paddings", None)

        preds = batch[DataKeys.PREDS]

        # preds: list of torch.Tensor(N, 5) as x1, y1, x2, y2, score
        preds = [preds[preds[:, 5] == batch_idx, :5] for batch_idx in range(len(preds))]
        preds = ff.utils.preprocess.adjust_results(preds, scales, paddings)
        batch[DataKeys.PREDS] = preds

        return batch


class FaceDetectionData(DataModule):
    input_transform_cls = FaceDetectionInputTransform
    output_transform_cls = FaceDetectionOutputTransform

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        **data_module_kwargs,
    ) -> "FaceDetectionData":
        return cls(
            FastFaceInput(RunningStage.TRAINING, train_dataset),
            FastFaceInput(RunningStage.VALIDATING, val_dataset),
            FastFaceInput(RunningStage.TESTING, test_dataset),
            FastFaceInput(RunningStage.PREDICTING, predict_dataset),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
            ),
            output_transform=cls.output_transform_cls(),
            **data_module_kwargs,
        )

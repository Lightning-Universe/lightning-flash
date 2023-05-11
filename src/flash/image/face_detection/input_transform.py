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

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

import torch

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _FASTFACE_AVAILABLE, _TORCHVISION_AVAILABLE

if _FASTFACE_AVAILABLE:
    import fastface as ff

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as T


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

        for i, (target, scale, padding) in enumerate(zip(targets, scales, paddings)):
            target["target_boxes"] *= scale
            target["target_boxes"][:, [0, 2]] += padding[0]
            target["target_boxes"][:, [1, 3]] += padding[1]
            targets[i]["target_boxes"] = target["target_boxes"]

        samples[DataKeys.TARGET] = targets
    samples[DataKeys.INPUT] = images

    return samples


@dataclass
class FaceDetectionInputTransform(InputTransform):
    def per_sample_transform(self) -> Callable:
        return T.Compose(
            [
                ApplyToKeys(DataKeys.INPUT, T.ToTensor()),
                ApplyToKeys(DataKeys.TARGET, ApplyToKeys("target_boxes", torch.as_tensor)),
            ]
        )

    def collate(self) -> Callable:
        return fastface_collate_fn

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
from enum import Enum
from typing import Dict, Optional, Tuple

import torch

from flash.data.process import Serializer
from flash.utils.imports import _KORNIA_AVAILABLE, _MATPLOTLIB_AVAILABLE

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None

if _KORNIA_AVAILABLE:
    import kornia as K
else:
    K = None


class SegmentationKeys(Enum):
    IMAGES = 'images'
    MASKS = 'masks'


class SegmentationLabels(Serializer):

    def __init__(self, labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None, visualize: bool = False):
        """A :class:`.Serializer` which converts the model outputs to the label of the argmax classification
        per pixel in the image for semantic segmentation tasks.

        Args:
            labels_map: A dictionary that map the labels ids to pixel intensities.
            visualise: Wether to visualise the image labels.
        """
        super().__init__()
        self.labels_map = labels_map
        self.visualize = visualize

    @staticmethod
    def labels_to_image(img_labels: torch.Tensor, labels_map: Dict[int, Tuple[int, int, int]]) -> torch.Tensor:
        """Function that given an image with labels ids and their pixels intrensity mapping,
           creates a RGB representation for visualisation purposes.
        """
        assert len(img_labels.shape) == 2, img_labels.shape
        H, W = img_labels.shape
        out = torch.empty(3, H, W, dtype=torch.uint8)
        for label_id, label_val in labels_map.items():
            mask = (img_labels == label_id)
            for i in range(3):
                out[i].masked_fill_(mask, label_val[i])
        return out

    @staticmethod
    def create_random_labels_map(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
        labels_map: Dict[int, Tuple[int, int, int]] = {}
        for i in range(num_classes):
            labels_map[i] = torch.randint(0, 255, (3, ))
        return labels_map

    def serialize(self, sample: torch.Tensor) -> torch.Tensor:
        assert len(sample.shape) == 3, sample.shape
        labels = torch.argmax(sample, dim=-3)  # HxW
        if self.visualize:
            if self.labels_map is None:
                # create random colors map
                num_classes = sample.shape[-3]
                labels_map = self.create_random_labels_map(num_classes)
            else:
                labels_map = self.labels_map
            labels_vis = self.labels_to_image(labels, labels_map)
            labels_vis = K.utils.tensor_to_image(labels_vis)
            plt.imshow(labels_vis)
            plt.show()
        return labels

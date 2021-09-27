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
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import torch

import flash
from flash.core.data.data_source import DefaultDataKeys, ImageLabelsMap
from flash.core.data.process import Serializer
from flash.core.utilities.imports import (
    _FIFTYONE_AVAILABLE,
    _KORNIA_AVAILABLE,
    _MATPLOTLIB_AVAILABLE,
    lazy_import,
    requires,
)

Segmentation = None
if _FIFTYONE_AVAILABLE:
    fol = lazy_import("fiftyone.core.labels")
    if TYPE_CHECKING:
        from fiftyone.core.labels import Segmentation
else:
    fol = None

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None

if _KORNIA_AVAILABLE:
    import kornia as K
else:
    K = None


class SegmentationLabels(Serializer):
    """A :class:`.Serializer` which converts the model outputs to the label of the argmax classification per pixel
    in the image for semantic segmentation tasks.

    Args:
        labels_map: A dictionary that map the labels ids to pixel intensities.
        visualize: Wether to visualize the image labels.
    """

    @requires("image")
    def __init__(self, labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None, visualize: bool = False):
        super().__init__()
        self.labels_map = labels_map
        self.visualize = visualize

    @staticmethod
    def labels_to_image(img_labels: torch.Tensor, labels_map: Dict[int, Tuple[int, int, int]]) -> torch.Tensor:
        """Function that given an image with labels ids and their pixels intrensity mapping, creates a RGB
        representation for visualisation purposes."""
        assert len(img_labels.shape) == 2, img_labels.shape
        H, W = img_labels.shape
        out = torch.empty(3, H, W, dtype=torch.uint8)
        for label_id, label_val in labels_map.items():
            mask = img_labels == label_id
            for i in range(3):
                out[i].masked_fill_(mask, label_val[i])
        return out

    @staticmethod
    def create_random_labels_map(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
        labels_map: Dict[int, Tuple[int, int, int]] = {}
        for i in range(num_classes):
            labels_map[i] = torch.randint(0, 255, (3,))
        return labels_map

    @requires("matplotlib")
    def _visualize(self, labels):
        if self.labels_map is None:
            self.labels_map = self.get_state(ImageLabelsMap).labels_map
        labels_vis = self.labels_to_image(labels, self.labels_map)
        labels_vis = K.utils.tensor_to_image(labels_vis)
        plt.imshow(labels_vis)
        plt.show()

    def serialize(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        preds = sample[DefaultDataKeys.PREDS]
        assert len(preds.shape) == 3, preds.shape
        labels = torch.argmax(preds, dim=-3)  # HxW

        if self.visualize and not flash._IS_TESTING:
            self._visualize(labels)
        return labels.tolist()


class FiftyOneSegmentationLabels(SegmentationLabels):
    """A :class:`.Serializer` which converts the model outputs to FiftyOne segmentation format.

    Args:
        labels_map: A dictionary that map the labels ids to pixel intensities.
        visualize: whether to visualize the image labels.
        return_filepath: Boolean determining whether to return a dict
            containing filepath and FiftyOne labels (True) or only a list of
            FiftyOne labels (False).
    """

    @requires("fiftyone")
    def __init__(
        self,
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
        visualize: bool = False,
        return_filepath: bool = False,
    ):
        super().__init__(labels_map=labels_map, visualize=visualize)

        self.return_filepath = return_filepath

    def serialize(self, sample: Dict[str, torch.Tensor]) -> Union[Segmentation, Dict[str, Any]]:
        labels = super().serialize(sample)
        fo_predictions = fol.Segmentation(mask=np.array(labels))
        if self.return_filepath:
            filepath = sample[DefaultDataKeys.METADATA]["filepath"]
            return {"filepath": filepath, "predictions": fo_predictions}
        return fo_predictions

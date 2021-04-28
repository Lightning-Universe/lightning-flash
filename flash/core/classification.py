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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# for visualisation
import kornia as K
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.utilities import rank_zero_warn

from flash.core.model import Task
from flash.data.process import ProcessState, Serializer


def binary_cross_entropy_with_logits(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calls BCE with logits and cast the target one_hot (y) encoding to floating point precision."""
    return F.binary_cross_entropy_with_logits(x, y.float())


@dataclass(unsafe_hash=True, frozen=True)
class ClassificationState(ProcessState):

    labels: Optional[List[str]]


class ClassificationTask(Task):

    def __init__(
        self,
        *args,
        loss_fn: Optional[Callable] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        multi_label: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
        **kwargs,
    ) -> None:
        if metrics is None:
            metrics = torchmetrics.Accuracy(subset_accuracy=multi_label)

        if loss_fn is None:
            loss_fn = binary_cross_entropy_with_logits if multi_label else F.cross_entropy
        super().__init__(
            *args,
            loss_fn=loss_fn,
            metrics=metrics,
            serializer=serializer or Classes(multi_label=multi_label),
            **kwargs,
        )

    def to_metrics_format(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self.hparams, "multi_label", False):
            return F.sigmoid(x)
        # we'll assume that the data always comes as `(B, C, ...)`
        return F.softmax(x, dim=1)


class ClassificationSerializer(Serializer):
    """A base class for classification serializers.

    Args:
        multi_label: If true, treats outputs as multi label logits.
    """

    def __init__(self, multi_label: bool = False):
        super().__init__()

        self._mutli_label = multi_label

    @property
    def multi_label(self) -> bool:
        return self._mutli_label


class Logits(ClassificationSerializer):
    """A :class:`.Serializer` which simply converts the model outputs (assumed to be logits) to a list."""

    def serialize(self, sample: Any) -> Any:
        return sample.tolist()


class Probabilities(ClassificationSerializer):
    """A :class:`.Serializer` which applies a softmax to the model outputs (assumed to be logits) and converts to a
    list."""

    def serialize(self, sample: Any) -> Any:
        if self.multi_label:
            return torch.sigmoid(sample).tolist()
        return torch.softmax(sample, -1).tolist()


class Classes(ClassificationSerializer):
    """A :class:`.Serializer` which applies an argmax to the model outputs (either logits or probabilities) and
    converts to a list.

    Args:
        multi_label: If true, treats outputs as multi label logits.

        threshold: The threshold to use for multi_label classification.
    """

    def __init__(self, multi_label: bool = False, threshold: float = 0.5):
        super().__init__(multi_label)

        self.threshold = threshold

    def serialize(self, sample: Any) -> Union[int, List[int]]:
        if self.multi_label:
            one_hot = (sample.sigmoid() > self.threshold).int().tolist()
            result = []
            for index, value in enumerate(one_hot):
                if value == 1:
                    result.append(index)
            return result
        return torch.argmax(sample, -1).tolist()


class Labels(Classes):
    """A :class:`.Serializer` which converts the model outputs (either logits or probabilities) to the label of the
    argmax classification.

    Args:
        labels: A list of labels, assumed to map the class index to the label for that class. If ``labels`` is not
            provided, will attempt to get them from the :class:`.ClassificationState`.

        multi_label: If true, treats outputs as multi label logits.

        threshold: The threshold to use for multi_label classification.
    """

    def __init__(self, labels: Optional[List[str]] = None, multi_label: bool = False, threshold: float = 0.5):
        super().__init__(multi_label=multi_label, threshold=threshold)
        self._labels = labels
        self.set_state(ClassificationState(labels))

    def serialize(self, sample: Any) -> Union[int, List[int], str, List[str]]:
        labels = None

        if self._labels is not None:
            labels = self._labels
        else:
            state = self.get_state(ClassificationState)
            if state is not None:
                labels = state.labels

        classes = super().serialize(sample)

        if labels is not None:
            if self.multi_label:
                return [labels[cls] for cls in classes]
            return labels[classes]
        else:
            rank_zero_warn(
                "No ClassificationState was found, this serializer will act as a Classes serializer.", UserWarning
            )
            return classes


class SegmentationLabels(Serializer):

    def __init__(self, labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None, visualise: bool = False):
        """A :class:`.Serializer` which converts the model outputs to the label of the argmax classification
        per pixel in the image for semantic segmentation tasks.

        Args:
            labels_map: A dictionary that map the labels ids to pixel intensities.
            visualise: Wether to visualise the image labels.
        """
        super().__init__()
        self.labels_map = labels_map
        self.visualise = visualise

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
        if self.visualise:
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

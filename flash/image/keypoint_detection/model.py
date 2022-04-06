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
from typing import Any, Dict, List, Optional

from flash.core.adapter import AdapterTask
from flash.core.registry import FlashRegistry
from flash.core.utilities.types import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE
from flash.image.keypoint_detection.backbones import KEYPOINT_DETECTION_HEADS


class KeypointDetector(AdapterTask):
    """The ``KeypointDetector`` is a :class:`~flash.Task` for detecting keypoints in images. For more details, see
    :ref:`keypoint_detection`.

    Args:
        num_keypoints: Number of keypoints to detect.
        num_classes: The number of keypoint classes.
        backbone: String indicating the backbone CNN architecture to use.
        head: String indicating the head module to use on top of the backbone.
        pretrained: Whether the model should be loaded with it's pretrained weights.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        learning_rate: The learning rate to use for training.
        predict_kwargs: dictionary containing parameters that will be used during the prediction phase.
        **kwargs: additional kwargs used for initializing the task
    """

    heads: FlashRegistry = KEYPOINT_DETECTION_HEADS

    required_extras: List[str] = ["image", "icevision"]

    def __init__(
        self,
        num_keypoints: int,
        num_classes: int = 2,
        backbone: Optional[str] = "resnet18_fpn",
        head: Optional[str] = "keypoint_rcnn",
        pretrained: bool = True,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        learning_rate: Optional[float] = None,
        predict_kwargs: Dict = None,
        **kwargs: Any,
    ):
        self.save_hyperparameters()

        predict_kwargs = predict_kwargs if predict_kwargs else {}
        metadata = self.heads.get(head, with_metadata=True)
        adapter = metadata["metadata"]["adapter"].from_task(
            self,
            num_keypoints=num_keypoints,
            num_classes=num_classes,
            backbone=backbone,
            head=head,
            pretrained=pretrained,
            predict_kwargs=predict_kwargs,
            **kwargs,
        )

        super().__init__(
            adapter,
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]) -> None:
        """This function is used only for debugging usage with CI."""
        # todo

    @property
    def predict_kwargs(self) -> Dict[str, Any]:
        """The kwargs used for the prediction step."""
        return self.adapter.predict_kwargs

    @predict_kwargs.setter
    def predict_kwargs(self, predict_kwargs: Dict[str, Any]):
        self.adapter.predict_kwargs = predict_kwargs

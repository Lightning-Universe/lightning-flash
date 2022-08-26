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
from typing import Any, Dict, List, Optional, Type, Union

from flash.core.adapter import AdapterTask
from flash.core.data.io.input import ServeInput
from flash.core.data.io.output import Output
from flash.core.integrations.icevision.transforms import IceVisionInputTransform
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.serve import Composition
from flash.core.utilities.imports import requires
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE, LR_SCHEDULER_TYPE, OPTIMIZER_TYPE
from flash.image.data import ImageDeserializer
from flash.image.detection.backbones import OBJECT_DETECTION_HEADS
from flash.image.detection.output import OBJECT_DETECTION_OUTPUTS


class ObjectDetector(AdapterTask):
    """The ``ObjectDetector`` is a :class:`~flash.Task` for detecting objects in images. For more details, see
    :ref:`object_detection`.

    Args:
        num_classes: The number of object classes.
        backbone: String indicating the backbone CNN architecture to use.
        head: String indicating the head module to use ontop of the backbone.
        pretrained: Whether the model should be loaded with it's pretrained weights.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        learning_rate: The learning rate to use for training.
        predict_kwargs: dictionary containing parameters that will be used during the prediction phase.
        kwargs: additional kwargs nessesary for initializing the backbone task
    """

    heads: FlashRegistry = OBJECT_DETECTION_HEADS
    outputs = Task.outputs + OBJECT_DETECTION_OUTPUTS

    required_extras: List[str] = ["image", "icevision", "effdet"]

    def __init__(
        self,
        num_classes: Optional[int] = None,
        labels: Optional[List[str]] = None,
        backbone: Optional[str] = "resnet18_fpn",
        head: Optional[str] = "retinanet",
        pretrained: bool = True,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        learning_rate: Optional[float] = None,
        predict_kwargs: Dict = None,
        **kwargs: Any,
    ):
        self.save_hyperparameters()

        if labels is not None and num_classes is None:
            num_classes = len(labels)

        self.labels = labels
        self.num_classes = num_classes

        predict_kwargs = predict_kwargs if predict_kwargs else {}
        metadata = self.heads.get(head, with_metadata=True)
        adapter = metadata["metadata"]["adapter"].from_task(
            self,
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

    @requires("serve")
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        sanity_check: bool = True,
        input_cls: Optional[Type[ServeInput]] = ImageDeserializer,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        output: Optional[Union[str, Output]] = None,
    ) -> Composition:
        return super().serve(host, port, sanity_check, input_cls, transform, transform_kwargs, output)

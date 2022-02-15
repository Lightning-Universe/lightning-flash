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
import warnings
from typing import Any, Dict, List, Optional

from flash.core.adapter import AdapterTask
from flash.core.data.io.input import DataKeys
from flash.core.data.transforms import ApplyToKeys
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _VISSL_AVAILABLE, requires
from flash.core.utilities.types import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE

if _VISSL_AVAILABLE:
    import classy_vision
    import classy_vision.generic.distributed_util

    from flash.image.embedding.backbones import IMAGE_EMBEDDER_BACKBONES
    from flash.image.embedding.strategies import IMAGE_EMBEDDER_STRATEGIES
    from flash.image.embedding.transforms import IMAGE_EMBEDDER_TRANSFORMS

    # patch this to avoid classy vision/vissl based distributed training
    classy_vision.generic.distributed_util.get_world_size = lambda: 1
else:
    IMAGE_EMBEDDER_BACKBONES = FlashRegistry("backbones")
    IMAGE_EMBEDDER_STRATEGIES = FlashRegistry("embedder_training_strategies")
    IMAGE_EMBEDDER_TRANSFORMS = FlashRegistry("embedder_transforms")


class ImageEmbedder(AdapterTask):
    """The ``ImageEmbedder`` is a :class:`~flash.Task` for obtaining feature vectors (embeddings) from images. For
    more details, see :ref:`image_embedder`.

    Args:
        training_strategy: Training strategy from VISSL,
            select between 'simclr', 'swav', 'dino', 'moco', or 'barlow_twins'.
        head: projection head used for task, select between
            'simclr_head', 'swav_head', 'dino_head', 'moco_head', or 'barlow_twins_head'.
        pretraining_transform: transform applied to input image for pre-training SSL model.
            Select between 'simclr_transform', 'swav_transform', 'dino_transform',
            'moco_transform', or 'barlow_twins_transform'.
        backbone: VISSL backbone, defaults to ``resnet``.
        pretrained: Use a pretrained backbone, defaults to ``False``.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        backbone_kwargs: arguments to be passed to VISSL backbones, i.e. ``vision_transformer`` and ``resnet``.
        training_strategy_kwargs: arguments passed to VISSL loss function, projection head and training hooks.
        pretraining_transform_kwargs: arguments passed to VISSL transforms.
    """

    training_strategies: FlashRegistry = IMAGE_EMBEDDER_STRATEGIES
    backbones: FlashRegistry = IMAGE_EMBEDDER_BACKBONES
    transforms: FlashRegistry = IMAGE_EMBEDDER_TRANSFORMS

    required_extras: List[str] = ["image", "vissl", "fairscale"]

    def __init__(
        self,
        training_strategy: str,
        head: str,
        pretraining_transform: str,
        backbone: str = "resnet",
        pretrained: bool = False,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        learning_rate: Optional[float] = None,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        training_strategy_kwargs: Optional[Dict[str, Any]] = None,
        pretraining_transform_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.save_hyperparameters()

        if backbone_kwargs is None:
            backbone_kwargs = {}

        if training_strategy_kwargs is None:
            training_strategy_kwargs = {}

        if pretraining_transform_kwargs is None:
            pretraining_transform_kwargs = {}

        backbone, _ = self.backbones.get(backbone)(pretrained=pretrained, **backbone_kwargs)

        metadata = self.training_strategies.get(training_strategy, with_metadata=True)
        loss_fn, head, hooks = metadata["fn"](head=head, **training_strategy_kwargs)

        adapter = metadata["metadata"]["adapter"].from_task(
            self,
            loss_fn=loss_fn,
            backbone=backbone,
            head=head,
            hooks=hooks,
        )

        super().__init__(
            adapter=adapter,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
        )

        input_transform, self.collate_fn = self.transforms.get(pretraining_transform)(**pretraining_transform_kwargs)
        self.input_transform = ApplyToKeys(DataKeys.INPUT, input_transform)

        warnings.warn(
            "Warning: VISSL ImageEmbedder overrides any user provided transforms"
            " with pre-defined transforms for the training strategy."
        )

    def on_train_start(self) -> None:
        self.adapter.on_train_start()

    def on_train_epoch_end(self) -> None:
        self.adapter.on_train_epoch_end()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.adapter.on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

    @classmethod
    @requires(["image", "vissl", "fairscale"])
    def available_training_strategies(cls) -> List[str]:
        registry: Optional[FlashRegistry] = getattr(cls, "training_strategies", None)
        if registry is None:
            return []
        return registry.available_keys()

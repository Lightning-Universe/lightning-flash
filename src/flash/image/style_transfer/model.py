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
from typing import Any, cast, List, NoReturn, Optional, Sequence, Tuple, Union

from torch import nn, Tensor

from flash.core.data.io.input import DataKeys
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _IMAGE_AVAILABLE
from flash.core.utilities.stability import beta
from flash.core.utilities.types import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE
from flash.image.style_transfer import STYLE_TRANSFER_BACKBONES

if _IMAGE_AVAILABLE:
    import pystiche.demo
    from pystiche import enc, loss
    from pystiche.image import read_image
else:

    class enc:
        Encoder = None
        MultiLayerEncoder = None

    class loss:
        class GramLoss:
            pass

        class PerceptualLoss:
            pass


from flash.image.style_transfer.utils import raise_not_supported

__all__ = ["StyleTransfer"]


@beta("Style transfer is currently in Beta.")
class StyleTransfer(Task):
    """``StyleTransfer`` is a :class:`~flash.Task` for transferring the style from one image onto another. For more
    details, see :ref:`style_transfer`.

    Args:
        style_image: Image or path to an image to derive the style from.
        model: The model by the style transfer task.
        backbone: A string or model to use to compute the style loss from.
        content_layer: Which layer from the backbone to extract the content loss from.
        content_weight: The weight associated with the content loss. A lower value will lose content over style.
        style_layers: Layers from the backbone to derive the style loss from.
        style_weight: The weight associated with the style loss. A lower value will lose style over content.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
    """

    backbones: FlashRegistry = STYLE_TRANSFER_BACKBONES

    required_extras: str = "image"

    def __init__(
        self,
        style_image: Optional[Union[str, Tensor]] = None,
        model: Optional[nn.Module] = None,
        backbone: str = "vgg16",
        content_layer: str = "relu2_2",
        content_weight: float = 1e5,
        style_layers: Union[Sequence[str], str] = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
        style_weight: float = 1e10,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        learning_rate: Optional[float] = None,
    ):
        self.save_hyperparameters(ignore="style_image")

        if style_image is None:
            style_image = self.default_style_image()
        elif isinstance(style_image, str):
            style_image = read_image(style_image)

        if model is None:
            model = pystiche.demo.transformer()

        if not isinstance(style_layers, (List, Tuple)):
            style_layers = (style_layers,)

        perceptual_loss = self._get_perceptual_loss(
            backbone=backbone,
            content_layer=content_layer,
            content_weight=content_weight,
            style_layers=style_layers,
            style_weight=style_weight,
        )
        perceptual_loss.set_style_image(style_image)

        super().__init__(
            model=model,
            loss_fn=perceptual_loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
        )

        self.perceptual_loss = perceptual_loss

    @staticmethod
    def default_style_image() -> Tensor:
        return pystiche.demo.images()["paint"].read(size=256)

    @staticmethod
    def _modified_gram_loss(encoder: enc.Encoder, *, score_weight: float) -> loss.GramLoss:
        # The official PyTorch examples as well as the reference implementation of the original author contain an
        # oversight: they normalize the representation twice by the number of channels. To be compatible with them, we
        # do the same here.
        class GramOperator(loss.GramLoss):
            def enc_to_repr(self, enc: Tensor) -> Tensor:
                rr = super().enc_to_repr(enc)
                num_channels = rr.size()[1]
                return rr / num_channels

        return GramOperator(encoder, score_weight=score_weight)

    def _get_perceptual_loss(
        self,
        *,
        backbone: str,
        content_layer: str,
        content_weight: float,
        style_layers: Sequence[str],
        style_weight: float,
    ) -> loss.PerceptualLoss:
        mle, _ = cast(enc.MultiLayerEncoder, self.backbones.get(backbone)())
        content_loss = loss.FeatureReconstructionLoss(mle.extract_encoder(content_layer), score_weight=content_weight)
        style_loss = loss.MultiLayerEncodingLoss(
            mle,
            style_layers,
            lambda encoder, layer_weight: self._modified_gram_loss(encoder, score_weight=layer_weight),
            layer_weights="sum",
            score_weight=style_weight,
        )
        return loss.PerceptualLoss(content_loss, style_loss)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        input_image = batch[DataKeys.INPUT]
        self.perceptual_loss.set_content_image(input_image)
        output_image = self(input_image)
        return self.perceptual_loss(output_image).total()

    def validation_step(self, batch: Any, batch_idx: int) -> NoReturn:
        raise_not_supported("validation")

    def test_step(self, batch: Any, batch_idx: int) -> NoReturn:
        raise_not_supported("test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        input_image = batch[DataKeys.INPUT]
        return self(input_image)

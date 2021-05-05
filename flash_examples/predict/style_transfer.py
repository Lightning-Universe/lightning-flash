import sys

import torch

from flash.utils.imports import _PYSTICHE_AVAILABLE

if _PYSTICHE_AVAILABLE:
    from pystiche import enc, loss, ops
else:
    print("Please, run `pip install pystiche`")
    sys.exit(0)

multi_layer_encoder = enc.vgg16_multi_layer_encoder()

content_layer = "relu2_2"
content_encoder = multi_layer_encoder.extract_encoder(content_layer)
content_weight = 1e5
content_loss = ops.FeatureReconstructionOperator(
    content_encoder, score_weight=content_weight
)


class GramOperator(ops.GramOperator):
    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        repr = super().enc_to_repr(enc)
        num_channels = repr.size()[1]
        return repr / num_channels


style_layers = ("relu1_2", "relu2_2", "relu3_3", "relu4_3")
style_weight = 1e10
style_loss = ops.MultiLayerEncodingOperator(
    multi_layer_encoder,
    style_layers,
    lambda encoder, layer_weight: GramOperator(encoder, score_weight=layer_weight),
    layer_weights="sum",
    score_weight=style_weight,
)

# TODO: this needs to be moved to the device to be trained on
# TODO: we need to register a style image here
perceptual_loss = loss.PerceptualLoss(content_loss, style_loss)


def loss_fn(image):
    perceptual_loss.set_content_image(image)
    return float(perceptual_loss(image))
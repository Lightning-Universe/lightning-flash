from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _PYSTICHE_AVAILABLE

STYLE_TRANSFER_BACKBONES = FlashRegistry("backbones")

__all__ = ["STYLE_TRANSFER_BACKBONES"]

if _PYSTICHE_AVAILABLE:

    from pystiche import enc

    for mle_fn in dir(enc):

        if not "multi_layer_encoder" in mle_fn:
            continue

        name = mle_fn.split("_")[0]

        STYLE_TRANSFER_BACKBONES(
            fn=lambda: (getattr(enc, mle_fn)(), None),
            name=mle_fn.split("_")[0],
            namespace="image/style_transfer",
            package="pystiche",
        )

import re

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _PYSTICHE_AVAILABLE

STYLE_TRANSFER_BACKBONES = FlashRegistry("backbones")

__all__ = ["STYLE_TRANSFER_BACKBONES"]

if _PYSTICHE_AVAILABLE:

    from pystiche import enc

    MLE_FN_PATTERN = re.compile(r"^(?P<name>\w+?)_multi_layer_encoder$")

    for mle_fn in dir(enc):
        match = MLE_FN_PATTERN.match(mle_fn)
        if not match:
            continue

        STYLE_TRANSFER_BACKBONES(
            fn=lambda: (getattr(enc, mle_fn)(), None),
            name=match.group("name"),
            namespace="image/style_transfer",
            package="pystiche",
        )

import re

from pystiche import enc

from flash.core.registry import FlashRegistry

__all__ = ["STYLE_TRANSFER_BACKBONES"]

MLE_FN_PATTERN = re.compile(r"^(?P<name>\w+?)_multi_layer_encoder$")

STYLE_TRANSFER_BACKBONES = FlashRegistry("backbones")

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

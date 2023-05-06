from flash.core.registry import ExternalRegistry, FlashRegistry
from flash.core.utilities.imports import _TOPIC_TEXT_AVAILABLE
from flash.core.utilities.providers import _HUGGINGFACE

if _TOPIC_TEXT_AVAILABLE:
    from transformers import AutoModel

    HUGGINGFACE_BACKBONES = ExternalRegistry(
        AutoModel.from_pretrained,
        "backbones",
        _HUGGINGFACE,
    )
else:
    HUGGINGFACE_BACKBONES = FlashRegistry("backbones")

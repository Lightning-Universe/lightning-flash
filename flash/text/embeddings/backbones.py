from flash.core.registry import ExternalRegistry, FlashRegistry
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.core.utilities.providers import _HUGGINGFACE

SENTENCE_TRANSFORMERS_BACKBONE = FlashRegistry("backbones")

if _TEXT_AVAILABLE:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    AUTOMODEL_BACKBONE = ExternalRegistry(
        AutoModel.from_pretrained,
        "backbones",
        _HUGGINGFACE,
    )
    AUTOTOKENIZER_BACKBONE = ExternalRegistry(
        AutoTokenizer.from_pretrained,
        "backbones",
        _HUGGINGFACE,
    )
    AUTOCONFIG_BACKBONE = ExternalRegistry(
        AutoConfig.from_pretrained,
        "backbones",
        _HUGGINGFACE,
    )

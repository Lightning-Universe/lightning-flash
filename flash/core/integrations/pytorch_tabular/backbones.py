import functools

from omegaconf import DictConfig, OmegaConf
from pytorch_tabular.config import OptimizerConfig

from flash.core.integrations.pytorch_tabular.adapter import PytorchTabularAdapter
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _PYTORCHTABULAR_AVAILABLE
from flash.core.utilities.providers import _PYTORCH_TABULAR

if _PYTORCHTABULAR_AVAILABLE:
    import pytorch_tabular.models as models
    from pytorch_tabular.models import (
        TabTransformerConfig,
        TabNetModelConfig,
        FTTransformerConfig,
        AutoIntConfig,
        NodeConfig,
        CategoryEmbeddingModelConfig,
)


PYTORCH_TABULAR_BACKBONES = FlashRegistry("backbones")


if _PYTORCHTABULAR_AVAILABLE:

    def load_pytorch_tabular(model_config, task_type, parameters: DictConfig, **kwargs):
        model_config = model_config(task=task_type, embedding_dims=parameters["embedding_dims"])
        model_callable = getattr(
            getattr(models, model_config._module_src), model_config._model_name
        )
        config = OmegaConf.merge(OmegaConf.create(parameters),
                                 model_config,
                                 OptimizerConfig())
        model = model_callable(config=config)
        return model

    for model_config, name in zip(
        [TabNetModelConfig, TabTransformerConfig, FTTransformerConfig, AutoIntConfig,
         NodeConfig, CategoryEmbeddingModelConfig],
        ["tabnet", "tabtransformer", "fttransformer", "autoint", "node", "category_embedding"],
    ):
        PYTORCH_TABULAR_BACKBONES(
            functools.partial(load_pytorch_tabular, model_config),
            name=name,
            providers=_PYTORCH_TABULAR,
            adapter=PytorchTabularAdapter,
        )

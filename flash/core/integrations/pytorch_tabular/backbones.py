import functools

from flash.core.integrations.pytorch_tabular.adapter import PytorchTabularAdapter
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _PYTORCHTABULAR_AVAILABLE
from flash.core.utilities.providers import _PYTORCH_TABULAR

if _PYTORCHTABULAR_AVAILABLE:
    from pytorch_tabular.models import (
        TabNetModel,
        TabTransformerModel,
        FTTransformerModel,
        AutoIntModel,
        NODEModel,
        CategoryEmbeddingModel
    )


PYTORCH_FORECASTING_BACKBONES = FlashRegistry("backbones")


if _PYTORCHTABULAR_AVAILABLE:

    def load_pytorch_tabular(model, **kwargs):
        pass
        #return model.from_dataset(time_series_dataset, **kwargs)

    for model, name in zip(
        [TabNetModel, TabTransformerModel, FTTransformerModel, AutoIntModel, NODEModel, CategoryEmbeddingModel],
        ["tabnet", "tabtransformer", "fttransformer", "autoint", "node", "category_embedding"],
    ):
        PYTORCH_FORECASTING_BACKBONES(
            functools.partial(load_pytorch_tabular(), model),
            name=name,
            providers=_PYTORCH_TABULAR,
            adapter=PytorchTabularAdapter,
        )

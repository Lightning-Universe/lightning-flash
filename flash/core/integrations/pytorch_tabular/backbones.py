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
import functools
import os
from typing import Callable, List, Optional, Union

import torchmetrics

from flash.core.integrations.pytorch_tabular.adapter import PytorchTabularAdapter
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _PYTORCHTABULAR_AVAILABLE
from flash.core.utilities.providers import _PYTORCH_TABULAR

if _PYTORCHTABULAR_AVAILABLE:
    import pytorch_tabular.models as models
    from omegaconf import DictConfig, OmegaConf
    from pytorch_tabular.config import ModelConfig
    from pytorch_tabular.models import (
        AutoIntConfig,
        CategoryEmbeddingModelConfig,
        FTTransformerConfig,
        NodeConfig,
        TabNetModelConfig,
        TabTransformerConfig,
    )


PYTORCH_TABULAR_BACKBONES = FlashRegistry("backbones")


if _PYTORCHTABULAR_AVAILABLE:

    def _read_parse_config(config, cls):
        if isinstance(config, str):
            if os.path.exists(config):
                _config = OmegaConf.load(config)
                if cls == ModelConfig:
                    cls = getattr(getattr(models, _config._module_src), _config._config_name)
                config = cls(
                    **{
                        k: v
                        for k, v in _config.items()
                        if (k in cls.__dataclass_fields__.keys()) and cls.__dataclass_fields__[k].init
                    }
                )
            else:
                raise ValueError(f"{config} is not a valid path")
        config = OmegaConf.structured(config)
        return config

    def load_pytorch_tabular(
        model_config_class,
        task_type,
        parameters: DictConfig,
        loss_fn: Callable,
        metrics: Optional[Union[torchmetrics.Metric, List[torchmetrics.Metric]]],
        **model_kwargs,
    ):
        model_config = model_config_class(task=task_type, embedding_dims=parameters["embedding_dims"], **model_kwargs)
        model_config = _read_parse_config(model_config, ModelConfig)
        model_callable = getattr(getattr(models, model_config._module_src), model_config._model_name)
        config = OmegaConf.merge(
            OmegaConf.create(parameters),
            OmegaConf.to_container(model_config),
        )
        model = model_callable(config=config, custom_loss=loss_fn, custom_metrics=metrics)
        return model

    for model_config_class, name in zip(
        [
            TabNetModelConfig,
            TabTransformerConfig,
            FTTransformerConfig,
            AutoIntConfig,
            NodeConfig,
            CategoryEmbeddingModelConfig,
        ],
        ["tabnet", "tabtransformer", "fttransformer", "autoint", "node", "category_embedding"],
    ):
        PYTORCH_TABULAR_BACKBONES(
            functools.partial(load_pytorch_tabular, model_config_class),
            name=name,
            providers=_PYTORCH_TABULAR,
            adapter=PytorchTabularAdapter,
        )

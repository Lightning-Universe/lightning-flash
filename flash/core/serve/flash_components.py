import inspect
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from flash.core.data.batch import _ServeInputProcessor
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys
from flash.core.data.io.output_transform import OutputTransform
from flash.core.serve import expose, ModelComponent
from flash.core.serve.types.base import BaseType
from flash.core.trainer import Trainer
from flash.core.utilities.stages import RunningStage


class FlashInputs(BaseType):
    def __init__(
        self,
        deserializer: Callable,
    ):
        self._deserializer = deserializer

    def serialize(self, *args) -> Any:  # pragma: no cover
        return None

    def deserialize(self, data: str) -> Any:  # pragma: no cover
        return self._deserializer(data)


class FlashOutputs(BaseType):
    def __init__(
        self,
        output: Callable,
    ):
        self._output = output

    def serialize(self, outputs) -> Any:  # pragma: no cover
        results = []
        if isinstance(outputs, (list, Tensor)):
            for output in outputs:
                result = self._output(output)
                if isinstance(result, Mapping):
                    result = result[DataKeys.PREDS]
                if isinstance(result, Tensor):
                    result = result.tolist()
                results.append(result)
        if len(results) == 1:
            return results[0]
        return results

    def deserialize(self, data: str) -> Any:  # pragma: no cover
        return None


def build_flash_serve_model_component(model, serve_input, output, transform, transform_kwargs):
    # TODO: Resolve this hack
    data_module = DataModule(
        predict_input=serve_input,
        batch_size=1,
        transform=transform,
        transform_kwargs=transform_kwargs,
    )

    class MockTrainer(Trainer):
        def __init__(self):
            super().__init__()
            self.state.stage = RunningStage.PREDICTING

        @property
        def lightning_module(self):
            return model

    data_module.trainer = MockTrainer()
    dataloader = data_module.predict_dataloader()

    collate_fn = dataloader.collate_fn

    class FlashServeModelComponent(ModelComponent):
        def __init__(self, model):
            self.model = model
            self.model.eval()
            self.serve_input = serve_input
            self.on_after_batch_transfer = data_module.on_after_batch_transfer
            self.output_transform = getattr(model, "_output_transform", None) or OutputTransform()
            # TODO (@tchaton) Remove this hack
            self.extra_arguments = len(inspect.signature(self.model.transfer_batch_to_device).parameters) == 3
            self.device = self.model.device

        @expose(
            inputs={"inputs": FlashInputs(_ServeInputProcessor(serve_input, collate_fn))},
            outputs={"outputs": FlashOutputs(output)},
        )
        def predict(self, inputs):
            with torch.no_grad():
                if self.extra_arguments:
                    inputs = self.model.transfer_batch_to_device(inputs, self.device, 0)
                else:
                    inputs = self.model.transfer_batch_to_device(inputs, self.device)
                inputs = self.on_after_batch_transfer(inputs, 0)
                preds = self.model.predict_step(inputs, 0)
                preds = self.output_transform(preds)
                return preds

    return FlashServeModelComponent(model)

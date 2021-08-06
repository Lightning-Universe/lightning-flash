import inspect
from typing import Any, Callable, Mapping

import torch
from pytorch_lightning.trainer.states import RunningStage

from flash.core.data.data_source import DefaultDataKeys
from flash.core.serve import expose, ModelComponent
from flash.core.serve.types.base import BaseType


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
        serializer: Callable,
    ):
        self._serializer = serializer

    def serialize(self, outputs) -> Any:  # pragma: no cover
        results = []
        if isinstance(outputs, (list, torch.Tensor)):
            for output in outputs:
                result = self._serializer(output)
                if isinstance(result, Mapping):
                    result = result[DefaultDataKeys.PREDS]
                results.append(result)
        if len(results) == 1:
            return results[0]
        return results

    def deserialize(self, data: str) -> Any:  # pragma: no cover
        return None


def build_flash_serve_model_component(model):

    data_pipeline = model.build_data_pipeline()

    class FlashServeModelComponent(ModelComponent):
        def __init__(self, model):
            self.model = model
            self.model.eval()
            self.data_pipeline = model.build_data_pipeline()
            self.worker_preprocessor = self.data_pipeline.worker_preprocessor(RunningStage.PREDICTING, is_serving=True)
            self.device_preprocessor = self.data_pipeline.device_preprocessor(RunningStage.PREDICTING)
            self.postprocessor = self.data_pipeline.postprocessor(RunningStage.PREDICTING, is_serving=True)
            # todo (tchaton) Remove this hack
            self.extra_arguments = len(inspect.signature(self.model.transfer_batch_to_device).parameters) == 3
            self.device = self.model.device

        @expose(
            inputs={"inputs": FlashInputs(data_pipeline.deserialize_processor())},
            outputs={"outputs": FlashOutputs(data_pipeline.serialize_processor())},
        )
        def predict(self, inputs):
            with torch.no_grad():
                inputs = self.worker_preprocessor(inputs)
                if self.extra_arguments:
                    inputs = self.model.transfer_batch_to_device(inputs, self.device, 0)
                else:
                    inputs = self.model.transfer_batch_to_device(inputs, self.device)
                inputs = self.device_preprocessor(inputs)
                preds = self.model.predict_step(inputs, 0)
                preds = self.postprocessor(preds)
                return preds

    return FlashServeModelComponent(model)

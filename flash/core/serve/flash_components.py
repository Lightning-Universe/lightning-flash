import inspect
from pathlib import Path
from typing import Any, Callable, Optional, Type

import torch
from pytorch_lightning.trainer.states import RunningStage

from flash import Task
from flash.core.serve import Composition, expose, GridModel, ModelComponent
from flash.core.serve.core import FilePath, GridModelValidArgs_T, GridserveScriptLoader
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

    def serialize(self, output) -> Any:  # pragma: no cover
        result = self._serializer(output)
        return result

    def deserialize(self, data: str) -> Any:  # pragma: no cover
        return None


class FlashServeScriptLoader(GridserveScriptLoader):

    model_cls: Optional[Task] = None

    def __init__(self, location: FilePath):
        self.location = location
        self.instance = self.model_cls.load_from_checkpoint(location)


class FlashServeModel(GridModel):

    def __init__(
        self,
        model_cls,
        *args: GridModelValidArgs_T,
        download_path: Optional[Path] = None,
        script_loader_cls: Type[GridserveScriptLoader] = FlashServeScriptLoader
    ):
        # todo (tchaton) Find a less hacky way.
        script_loader_cls.model_cls = model_cls
        super().__init__(*args, download_path=download_path, script_loader_cls=script_loader_cls)

    def to_flash_component(self):
        model = self.instance.instance

        class FlashServeModelComponent(ModelComponent):

            def __init__(self, model):
                self.model = model
                self.model.eval()
                self.data_pipeline = self.model.build_data_pipeline()
                self.worker_preprocessor = self.data_pipeline.worker_preprocessor(
                    RunningStage.PREDICTING, is_serving=True
                )
                self.device_preprocessor = self.data_pipeline.device_preprocessor(RunningStage.PREDICTING)
                self.postprocessor = self.data_pipeline.postprocessor(RunningStage.PREDICTING, is_serving=True)
                # todo (tchaton) Remove this hack
                self.extra_arguments = len(inspect.signature(self.model.transfer_batch_to_device).parameters) == 3

            @expose(
                inputs={"inputs": FlashInputs(model.data_pipeline.deserialize_processor())},
                outputs={"outputs": FlashOutputs(model.data_pipeline.serialize_processor())},
            )
            def predict(self, inputs):
                with torch.no_grad():
                    inputs = self.worker_preprocessor(inputs)
                    if self.extra_arguments:
                        inputs = self.model.transfer_batch_to_device(inputs, model.device, 0)
                    else:
                        inputs = self.model.transfer_batch_to_device(inputs, model.device)
                    inputs = self.device_preprocessor(inputs)
                    preds = self.model.predict_step(inputs, 0)
                    preds = self.postprocessor(preds)
                    return preds

        comp = FlashServeModelComponent(model)
        return comp

    def serve(self, host: str = "127.0.0.1", port: int = 8000):
        comp = self.to_flash_component()
        composition = Composition(predict=comp)
        composition.serve(host=host, port=port)

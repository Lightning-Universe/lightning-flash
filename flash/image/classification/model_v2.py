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
from typing import Mapping, Optional, Tuple, Union

import torch

from flash.core.data.process import (
    Deserializer,
    DeserializerMapping,
    Postprocess,
    Preprocess,
    Serializer,
    SerializerMapping,
)
from flash.core.data_v2.data_pipeline import DataPipeline, DataPipelineState
from flash.core.registry import FlashRegistry
from flash.image.classification.model import ImageClassifier


class ImageClassifier(ImageClassifier):

    _flash_datasets_registry: Optional[FlashRegistry]

    @staticmethod
    def _resolve(
        old_deserializer: Optional[Deserializer],
        old_postprocess: Optional[Postprocess],
        old_serializer: Optional[Serializer],
        new_deserializer: Optional[Deserializer],
        new_postprocess: Optional[Postprocess],
        new_serializer: Optional[Serializer],
    ) -> Tuple[Optional[Deserializer], Optional[Preprocess], Optional[Postprocess], Optional[Serializer]]:
        """Resolves the correct :class:`~flash.core.data.process.Preprocess`, :class:`~flash.core.data.process.Postprocess`, and
        :class:`~flash.core.data.process.Serializer` to use, choosing ``new_*`` if it is not None or a base class
        (:class:`~flash.core.data.process.Preprocess`, :class:`~flash.core.data.process.Postprocess`, or
        :class:`~flash.core.data.process.Serializer`) and ``old_*`` otherwise.

        Args:
            old_preprocess: :class:`~flash.core.data.process.Preprocess` to be overridden.
            old_postprocess: :class:`~flash.core.data.process.Postprocess` to be overridden.
            old_serializer: :class:`~flash.core.data.process.Serializer` to be overridden.
            new_preprocess: :class:`~flash.core.data.process.Preprocess` to override with.
            new_postprocess: :class:`~flash.core.data.process.Postprocess` to override with.
            new_serializer: :class:`~flash.core.data.process.Serializer` to override with.

        Returns:
            The resolved :class:`~flash.core.data.process.Preprocess`, :class:`~flash.core.data.process.Postprocess`,
            and :class:`~flash.core.data.process.Serializer`.
        """
        preprocess = None
        deserializer = old_deserializer
        if new_deserializer is not None and type(new_deserializer) != Deserializer:
            deserializer = new_deserializer

        postprocess = old_postprocess
        if new_postprocess is not None and type(new_postprocess) != Postprocess:
            postprocess = new_postprocess

        serializer = old_serializer
        if new_serializer is not None and type(new_serializer) != Serializer:
            serializer = new_serializer

        return deserializer, preprocess, postprocess, serializer

    @torch.jit.unused
    @property
    def deserializer(self) -> Optional[Deserializer]:
        return self._deserializer

    @deserializer.setter
    def deserializer(self, deserializer: Union[Deserializer, Mapping[str, Deserializer]]):
        if isinstance(deserializer, Mapping):
            deserializer = DeserializerMapping(deserializer)
        self._deserializer = deserializer

    @torch.jit.unused
    @property
    def serializer(self) -> Optional[Serializer]:
        """The current :class:`.Serializer` associated with this model.

        If this property was set to a mapping
        (e.g. ``.serializer = {'output1': SerializerOne()}``) then this will be a :class:`.MappingSerializer`.
        """
        return self._serializer

    @torch.jit.unused
    @serializer.setter
    def serializer(self, serializer: Union[Serializer, Mapping[str, Serializer]]):
        if isinstance(serializer, Mapping):
            serializer = SerializerMapping(serializer)
        self._serializer = serializer

    def build_data_pipeline(
        self,
        data_source: Optional[str] = None,
        deserializer: Optional[Deserializer] = None,
        data_pipeline: Optional[DataPipeline] = None,
    ) -> Optional[DataPipeline]:
        """Build a :class:`.DataPipeline` incorporating available
        :class:`~flash.core.data.process.Preprocess` and :class:`~flash.core.data.process.Postprocess`
        objects. These will be overridden in the following resolution order (lowest priority first):

        - Lightning ``Datamodule``, either attached to the :class:`.Trainer` or to the :class:`.Task`.
        - :class:`.Task` defaults given to :meth:`.Task.__init__`.
        - :class:`.Task` manual overrides by setting :py:attr:`~data_pipeline`.
        - :class:`.DataPipeline` passed to this method.

        Args:
            data_source: A string that indicates the format of the data source to use which will override
                the current data source format used.
            deserializer: deserializer to use
            data_pipeline: Optional highest priority source of
                :class:`~flash.core.data.process.Preprocess` and :class:`~flash.core.data.process.Postprocess`.

        Returns:
            The fully resolved :class:`.DataPipeline`.
        """
        deserializer, old_data_source, preprocess, postprocess, serializer = None, None, None, None, None

        # Datamodule
        datamodule = None
        if self.trainer is not None and hasattr(self.trainer, "datamodule"):
            datamodule = self.trainer.datamodule
        elif getattr(self, "datamodule", None) is not None:
            datamodule = self.datamodule

        datamodule.data_pipeline

        if getattr(datamodule, "data_pipeline", None) is not None:
            old_data_source = getattr(datamodule.data_pipeline, "data_source", None)
            preprocess = getattr(datamodule.data_pipeline, "_preprocess_pipeline", None)
            postprocess = getattr(datamodule.data_pipeline, "_postprocess_pipeline", None)
            serializer = getattr(datamodule.data_pipeline, "_serializer", None)
            deserializer = getattr(datamodule.data_pipeline, "_deserializer", None)

        # Defaults / task attributes
        deserializer, preprocess, postprocess, serializer = self._resolve(
            deserializer,
            preprocess,
            postprocess,
            serializer,
            self._deserializer,
            self._preprocess,
            self._postprocess,
            self._serializer,
        )

        # Datapipeline
        if data_pipeline is not None:
            deserializer, preprocess, postprocess, serializer = self._resolve(
                deserializer,
                preprocess,
                postprocess,
                serializer,
                getattr(data_pipeline, "_deserializer", None),
                getattr(data_pipeline, "_preprocess_pipeline", None),
                getattr(data_pipeline, "_postprocess_pipeline", None),
                getattr(data_pipeline, "_serializer", None),
            )

        data_source = data_source or old_data_source

        if deserializer is None or type(deserializer) is Deserializer:
            deserializer = getattr(preprocess, "deserializer", deserializer)

        data_pipeline = DataPipeline(None, postprocess, deserializer, serializer)
        self._data_pipeline_state = self._data_pipeline_state or DataPipelineState()
        self.attach_data_pipeline_state(self._data_pipeline_state)
        self._data_pipeline_state = data_pipeline.initialize(self._data_pipeline_state)
        return data_pipeline

    @torch.jit.unused
    @property
    def data_pipeline(self) -> DataPipeline:
        """The current :class:`.DataPipeline`.

        If set, the new value will override the :class:`.Task` defaults. See
        :py:meth:`~build_data_pipeline` for more details on the resolution order.
        """
        return self.build_data_pipeline()

    @torch.jit.unused
    @data_pipeline.setter
    def data_pipeline(self, data_pipeline: Optional[DataPipeline]) -> None:
        self._deserializer, self._preprocess, self._postprocess, self.serializer = self._resolve(
            self._deserializer,
            self._preprocess,
            self._postprocess,
            self._serializer,
            getattr(data_pipeline, "_deserializer", None),
            getattr(data_pipeline, "_preprocess_pipeline", None),
            getattr(data_pipeline, "_postprocess_pipeline", None),
            getattr(data_pipeline, "_serializer", None),
        )

        # self._preprocess.state_dict()
        if getattr(self._preprocess, "_ddp_params_and_buffers_to_ignore", None):
            self._ddp_params_and_buffers_to_ignore = self._preprocess._ddp_params_and_buffers_to_ignore

    @torch.jit.unused
    @property
    def preprocess(self) -> Preprocess:
        return getattr(self.data_pipeline, "_preprocess_pipeline", None)

    @torch.jit.unused
    @property
    def postprocess(self) -> Postprocess:
        return getattr(self.data_pipeline, "_postprocess_pipeline", None)

    def on_train_dataloader(self) -> None:
        pass

    def on_val_dataloader(self) -> None:
        pass

    def on_test_dataloader(self, *_) -> None:
        pass

    def on_predict_dataloader(self) -> None:
        pass

    def on_predict_end(self) -> None:
        pass

    def on_fit_end(self) -> None:
        pass

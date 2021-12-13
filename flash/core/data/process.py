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
from typing import Any, Mapping
from warnings import warn

from deprecate import deprecated

import flash
from flash.core.data.io.input import ServeInput as Deserializer
from flash.core.data.io.output import Output


class DeserializerMapping(Deserializer):
    # TODO: This is essentially a duplicate of OutputMapping, should be abstracted away somewhere
    """Deserializer Mapping."""

    def __init__(self, deserializers: Mapping[str, Deserializer]):
        super().__init__()

        self._deserializers = deserializers

    def deserialize(self, sample: Any) -> Any:
        if isinstance(sample, Mapping):
            return {key: deserializer.deserialize(sample[key]) for key, deserializer in self._deserializers.items()}
        raise ValueError("The model output must be a mapping when using a DeserializerMapping.")

    def attach_data_pipeline_state(self, data_pipeline_state: "flash.core.data.data_pipeline.DataPipelineState"):
        for deserializer in self._deserializers.values():
            deserializer.attach_data_pipeline_state(data_pipeline_state)


class Serializer(Output):
    """Deprecated.

    Use ``Output`` instead.
    """

    @deprecated(
        None,
        "0.6.0",
        "0.7.0",
        template_mgs="`Serializer` was deprecated in v%(deprecated_in)s in favor of `Output`. "
        "It will be removed in v%(remove_in)s.",
        stream=functools.partial(warn, category=FutureWarning),
    )
    def __init__(self):
        super().__init__()
        self._is_enabled = True

    @staticmethod
    @deprecated(
        None,
        "0.6.0",
        "0.7.0",
        template_mgs="`Serializer` was deprecated in v%(deprecated_in)s in favor of `Output`. "
        "It will be removed in v%(remove_in)s.",
        stream=functools.partial(warn, category=FutureWarning),
    )
    def serialize(sample: Any) -> Any:
        """Deprecated.

        Use ``Output.transform`` instead.
        """
        return sample

    def transform(self, sample: Any) -> Any:
        return self.serialize(sample)

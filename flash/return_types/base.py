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
from typing import Any, Optional

from flash.data.data_pipeline import DataPipeline
from flash.data.process import Postprocess


class ReturnType:

    def convert(self, sample: Any) -> Any:
        raise NotImplementedError

    def wrap(self, postprocess: Optional[Postprocess]) -> Postprocess:
        if postprocess is None:
            postprocess = Postprocess()

        return _ReturnTypePostprocessWrapper(self, postprocess)


class _ReturnTypePostprocessWrapper(Postprocess):
    def __init__(self, return_type: ReturnType, postprocess: Postprocess):
        super().__init__()

        self._postprocess = postprocess
        self._return_type = return_type

    def _resolve_and_run(self, function_name: str, *args):
        self._postprocess.running_stage = self.running_stage
        self._postprocess.current_fn = self.current_fn
        function_name = DataPipeline._resolve_function_hierarchy(
            function_name,
            self._postprocess,
            self.running_stage,
            object_type=Postprocess,
        )
        return getattr(self._postprocess, function_name)(*args)

    def per_batch_transform(self, batch: Any) -> Any:
        return self._resolve_and_run("per_batch_transform", batch)

    def uncollate(self, batch: Any) -> Any:
        return self._resolve_and_run("uncollate", batch)

    def per_sample_transform(self, sample: Any) -> Any:
        sample = self._resolve_and_run("per_sample_transform", sample)
        return self._return_type.convert(sample)

    def save_sample(self, sample: Any, path: str) -> None:
        self._resolve_and_run("save_sample", sample, path)

    def save_data(self, data: Any, path: str) -> None:
        self._resolve_and_run("save_data", data, path)

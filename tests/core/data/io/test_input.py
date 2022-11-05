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
import pytest

from flash.core.data.io.input import Input, IterableInput, ServeInput
from flash.core.utilities.imports import _CORE_TESTING
from flash.core.utilities.stages import RunningStage


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_input_validation():
    with pytest.raises(RuntimeError, match="Use `IterableInput` instead."):

        class InvalidInput(Input):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.data = iter([1, 2, 3])

        InvalidInput(RunningStage.TRAINING)

    class ValidInput(Input):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.data = [1, 2, 3]

    ValidInput(RunningStage.TRAINING)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_iterable_input_validation():
    with pytest.raises(RuntimeError, match="Use `Input` instead."):

        class InvalidIterableInput(IterableInput):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.data = [1, 2, 3]

        InvalidIterableInput(RunningStage.TRAINING)

    class ValidIterableInput(IterableInput):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.data = iter([1, 2, 3])

    ValidIterableInput(RunningStage.TRAINING)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_serve_input():
    server_input = ServeInput()
    assert server_input.serving
    with pytest.raises(NotImplementedError):
        server_input._call_load_sample("")

    class CustomServeInput(ServeInput):
        def serve_load_data(self, data):
            raise NotImplementedError

        def serve_load_sample(self, data):
            return data + 1

    with pytest.raises(TypeError, match="serve_load_data"):
        serve_input = CustomServeInput()

    class CustomServeInput2(ServeInput):
        def serve_load_sample(self, data):
            return data + 1

    serve_input = CustomServeInput2()
    assert serve_input._call_load_sample(1) == 2

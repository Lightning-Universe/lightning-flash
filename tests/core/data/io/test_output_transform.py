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
import torch

from flash.core.data.batch import default_uncollate
from flash.core.data.io.output_transform import _OutputTransformProcessor


def test_output_transform_processor_str():
    output_transform_processor = _OutputTransformProcessor(
        default_uncollate,
        torch.relu,
        torch.softmax,
        None,
    )
    assert str(output_transform_processor) == (
        "_OutputTransformProcessor:\n"
        "\t(per_batch_transform): FuncModule(relu)\n"
        "\t(uncollate_fn): FuncModule(default_uncollate)\n"
        "\t(per_sample_transform): FuncModule(softmax)\n"
        "\t(output): None"
    )

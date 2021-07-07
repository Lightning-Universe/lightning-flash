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
import os
import re

import pytest
import torch

from flash.core.utilities.imports import _IMAGE_AVAILABLE
from flash.image import ImageEmbedder
from tests.helpers.utils import _IMAGE_TESTING


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
@pytest.mark.parametrize("jitter, args", [(torch.jit.script, ()), (torch.jit.trace, (torch.rand(1, 3, 32, 32), ))])
def test_jit(tmpdir, jitter, args):
    path = os.path.join(tmpdir, "test.pt")

    model = ImageEmbedder(embedding_dim=128)
    model.eval()

    model = jitter(model, *args)

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    out = model(torch.rand(1, 3, 32, 32))
    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 128])


@pytest.mark.skipif(_IMAGE_AVAILABLE, reason="image libraries are installed.")
def test_load_from_checkpoint_dependency_error():
    with pytest.raises(ModuleNotFoundError, match=re.escape("'lightning-flash[image]'")):
        ImageEmbedder.load_from_checkpoint("not_a_real_checkpoint.pt")

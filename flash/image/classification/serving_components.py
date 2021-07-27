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
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict

import flash
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Deserializer
from flash.core.utilities.imports import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE, requires_extras

if _TORCHVISION_AVAILABLE:
    import torchvision

if _PIL_AVAILABLE:
    from PIL import Image as PILImage
else:

    class Image:
        Image = None


class ImageDeserializer(Deserializer):

    @requires_extras("image")
    def __init__(self):
        super().__init__()
        self.to_tensor = torchvision.transforms.ToTensor()

    def deserialize(self, data: str) -> Dict:
        encoded_with_padding = (data + "===").encode("ascii")
        img = base64.b64decode(encoded_with_padding)
        buffer = BytesIO(img)
        img = PILImage.open(buffer, mode="r")
        return {
            DefaultDataKeys.INPUT: img,
        }

    @property
    def example_input(self) -> str:
        with (Path(flash.ASSETS_ROOT) / "fish.jpg").open("rb") as f:
            return base64.b64encode(f.read()).decode("UTF-8")

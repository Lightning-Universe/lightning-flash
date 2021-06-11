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
import sys

from flash.core.data.utils import download_data
from flash.core.utilities.imports import _KORNIA_AVAILABLE, _PYTORCHVIDEO_AVAILABLE
from flash.video import VideoClassifier

if not (_PYTORCHVIDEO_AVAILABLE and _KORNIA_AVAILABLE):
    print("Please, run `pip install torchvideo kornia`")
    sys.exit(0)

# 1. Download a video clip dataset. Find more dataset at https://pytorchvideo.readthedocs.io/en/latest/data.html
download_data("https://pl-flash-data.s3.amazonaws.com/kinetics.zip")

model = VideoClassifier.load_from_checkpoint(
    "https://flash-weights.s3.amazonaws.com/video_classification.pt", pretrained=False
)

# 2. Make a prediction
predictions = model.predict("data/kinetics/predict/")
print(predictions)

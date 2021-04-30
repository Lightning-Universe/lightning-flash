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
import flash
from flash.data.utils import download_data
from flash.vision import SemanticSegmentation
from flash.vision.segmentation.serialization import SegmentationLabels

# 1. Download the data
# This is a Dataset with Semantic Segmentation Labels generated via CARLA self-driving simulator.
# The data was generated as part of the Lyft Udacity Challenge.
# More info here: https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge
download_data(
    "https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip", "data/"
)

# 2. Load the model from a checkpoint
model = SemanticSegmentation.load_from_checkpoint(
    "https://flash-weights.s3.amazonaws.com/semantic_segmentation_model.pt"
)
model.serializer = SegmentationLabels(visualize=True)

# 3. Predict what's on a few images and visualize!
predictions = model.predict([
    'data/CameraRGB/F61-1.png',
    'data/CameraRGB/F62-1.png',
    'data/CameraRGB/F63-1.png',
])

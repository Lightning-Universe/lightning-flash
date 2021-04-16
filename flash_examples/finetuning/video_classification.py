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

import torch
from torch.utils.data import SequentialSampler

import flash
from flash.data.utils import download_data
from flash.utils.imports import _KORNIA_AVAILABLE, _PYTORCHVIDEO_AVAILABLE
from flash.video import VideoClassificationData, VideoClassifier

if _PYTORCHVIDEO_AVAILABLE and _KORNIA_AVAILABLE:
    import kornia.augmentation as K
    from pytorchvideo.transforms import ApplyTransformToKey, RandomShortSideScale, UniformTemporalSubsample
    from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip
else:
    print("Please, run `pip install torchvideo kornia`")
    sys.exit(0)

# 1. Download a video dataset: https://pytorchvideo.readthedocs.io/en/latest/data.html
download_data("NEED_TO_BE_CREATED")

# 2. [Optional] Specify transforms to be used during training.
train_transform = {
    "post_tensor_transform": Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(8),
                RandomShortSideScale(min_size=256, max_size=320),
                RandomCrop(244),
                RandomHorizontalFlip(p=0.5),
            ]),
        ),
    ]),
    "per_batch_transform_on_device": Compose([
        ApplyTransformToKey(
            key="video",
            transform=K.VideoSequential(
                K.Normalize(torch.tensor([0.45, 0.45, 0.45]), torch.tensor([0.225, 0.225, 0.225])),
                K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
                data_format="BCTHW",
                same_on_frame=False
            )
        ),
    ]),
}

# 3. Load the data
datamodule = VideoClassificationData.from_paths(
    train_data_path="path_to_train_data",
    clip_sampler="uniform",
    clip_duration=2,
    video_sampler=SequentialSampler,
    decode_audio=False,
    train_transform=train_transform
)

# 4. List the available models
print(VideoClassifier.available_models())
# out: ['efficient_x3d_s', 'efficient_x3d_xs', 'slow_r50', 'slowfast_r101', 'slowfast_r50', 'x3d_m', 'x3d_s', 'x3d_xs']

# 5. Build the model
model = VideoClassifier(num_classes=datamodule.num_classes, pretrained=False)

# 6. Train the model
trainer = flash.Trainer(fast_dev_run=True)

# 6. Finetune the model
trainer.finetune(model, datamodule=datamodule)

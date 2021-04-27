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
import sys
from typing import Callable, List

import torch
from torch.utils.data import SequentialSampler

import flash
from flash.core.classification import Labels
from flash.data.utils import download_data
from flash.core.finetuning import NoFreeze
from flash.utils.imports import _KORNIA_AVAILABLE, _PYTORCHVIDEO_AVAILABLE
from flash.video import VideoClassificationData, VideoClassifier

if _PYTORCHVIDEO_AVAILABLE and _KORNIA_AVAILABLE:
    import kornia.augmentation as K
    from pytorchvideo.transforms import ApplyTransformToKey, RandomShortSideScale, UniformTemporalSubsample
    from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip
else:
    print("Please, run `pip install torchvideo kornia`")
    sys.exit(0)

if __name__ == '__main__':

    _PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 1. Download a video clip dataset. Find more dataset at https://pytorchvideo.readthedocs.io/en/latest/data.html
    download_data("https://pl-flash-data.s3.amazonaws.com/kinetics.zip")

    # 2. [Optional] Specify transforms to be used during training.
    # Flash helps you to place your transform exactly where you want.
    # Learn more at https://lightning-flash.readthedocs.io/en/latest/general/data.html#flash.data.process.Preprocess
    post_tensor_transform = [UniformTemporalSubsample(8), RandomShortSideScale(min_size=256, max_size=320), RandomCrop(244)]
    per_batch_transform_on_device = [K.Normalize(torch.tensor([0.45, 0.45, 0.45]), torch.tensor([0.225, 0.225, 0.225]))]

    train_post_tensor_transform = post_tensor_transform + [RandomHorizontalFlip(p=0.5)]
    train_per_batch_transform_on_device = per_batch_transform_on_device


    def make_transform(
        post_tensor_transform: List[Callable] = post_tensor_transform,
        per_batch_transform_on_device: List[Callable] = per_batch_transform_on_device
    ):
        return {
            "post_tensor_transform": Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(post_tensor_transform),
                ),
            ]),
            "per_batch_transform_on_device": Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=K.VideoSequential(*per_batch_transform_on_device, data_format="BCTHW", same_on_frame=False)
                ),
            ]),
        }

    # 3. Load the data from directories.
    datamodule = VideoClassificationData.from_paths(
        train_data_path=os.path.join(_PATH_ROOT, "data/kinetics/train"),
        val_data_path=os.path.join(_PATH_ROOT, "data/kinetics/val"),
        predict_data_path=os.path.join(_PATH_ROOT, "data/kinetics/predict"),
        clip_sampler="uniform",
        clip_duration=1,
        video_sampler=SequentialSampler,
        decode_audio=False,
        train_transform=make_transform(train_post_tensor_transform),
        val_transform=make_transform(),
        predict_transform=make_transform(),
        num_workers=4,
    )

    # 4. List the available models
    print(VideoClassifier.available_models())
    # out: ['efficient_x3d_s', 'efficient_x3d_xs', ... ,slowfast_r50', 'x3d_m', 'x3d_s', 'x3d_xs']
    print(VideoClassifier.get_model_details("x3d_xs"))

    # 5. Build the model - `x3d_xs` comes with `nn.Softmax` by default for their `head_activation`.
    model = VideoClassifier(model="x3d_xs", num_classes=datamodule.num_classes)
    model.serializer = Labels()

    # 6. Finetune the model
    trainer = flash.Trainer(max_epochs=20, gpus=2, accelerator="ddp")
    trainer.finetune(model, datamodule=datamodule, strategy=NoFreeze())

    #trainer.save_checkpoint("video_classification.pt")
    #model = VideoClassifier.load_from_checkpoint("video_classification.pt")

    # 7. Make a prediction
    val_folder = os.path.join(_PATH_ROOT, os.path.join(_PATH_ROOT, "data/kinetics/predict"))
    predictions = model.predict([os.path.join(val_folder, f) for f in os.listdir(val_folder)])
    print(predictions)

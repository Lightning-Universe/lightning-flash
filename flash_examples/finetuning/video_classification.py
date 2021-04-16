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

download_data("NEED_TO_BE_CREATED")

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

datamodule = VideoClassificationData.from_paths(
    train_folder="data/action_youtube_naudio",
    clip_sampler="uniform",
    clip_duration=2,
    video_sampler=SequentialSampler,
    decode_audio=False,
    train_transform=train_transform
)

print(VideoClassifier.available_models())

model = VideoClassifier(num_classes=datamodule.num_classes, pretrained=False)

trainer = flash.Trainer(fast_dev_run=True)

trainer.finetune(model, datamodule=datamodule)

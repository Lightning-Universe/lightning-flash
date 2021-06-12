import os
import sys
from typing import Callable, List
import itertools

import fiftyone as fo
import fiftyone.zoo as foz

from flash import Trainer
from flash.core.classification import FiftyOneLabels
from flash.core.data.utils import download_data
from flash.core.finetuning import NoFreeze
from flash.video import VideoClassificationData, VideoClassifier

import torch
from torch.utils.data.sampler import RandomSampler
import kornia.augmentation as K
from pytorchvideo.transforms import ApplyTransformToKey, RandomShortSideScale, UniformTemporalSubsample
from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip

# 1. Load your FiftyOne dataset
# Find more dataset at https://pytorchvideo.readthedocs.io/en/latest/data.html

train_dataset = fo.Dataset.from_dir(
    dataset_dir="data/kinetics/train",
    dataset_type=fo.types.VideoClassificationDirectoryTree,
)

val_dataset = fo.Dataset.from_dir(
    dataset_dir="data/kinetics/val",
    dataset_type=fo.types.VideoClassificationDirectoryTree,
)

predict_dataset = fo.Dataset.from_dir(
    dataset_dir="data/kinetics/predict",
    dataset_type=fo.types.VideoDirectory,
)

# 2. [Optional] Specify transforms to be used during training.
# Flash helps you to place your transform exactly where you want.
# Learn more at:
# https://lightning-flash.readthedocs.io/en/latest/general/data.html#flash.core.data.process.Preprocess
post_tensor_transform = [UniformTemporalSubsample(8), RandomShortSideScale(min_size=256, max_size=320)]
per_batch_transform_on_device = [K.Normalize(torch.tensor([0.45, 0.45, 0.45]), torch.tensor([0.225, 0.225, 0.225]))]

train_post_tensor_transform = post_tensor_transform + [RandomCrop(244), RandomHorizontalFlip(p=0.5)]
val_post_tensor_transform = post_tensor_transform + [CenterCrop(244)]
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
                transform=K.VideoSequential(
                    *per_batch_transform_on_device, data_format="BCTHW", same_on_frame=False
                )
            ),
        ]),
    }


# 2. Load the Datamodule
datamodule = VideoClassificationData.from_fiftyone_datasets(
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    predict_dataset = predict_dataset,
    label_field = "ground_truth",
    train_transform=make_transform(train_post_tensor_transform),
    val_transform=make_transform(val_post_tensor_transform),
    predict_transform=make_transform(val_post_tensor_transform),
    batch_size=8,
    clip_sampler="uniform",
    clip_duration=1,
    video_sampler=RandomSampler,
    decode_audio=False,
    num_workers=8
)

# 3. Build the model
model = VideoClassifier(
    backbone="x3d_xs",
    num_classes=datamodule.num_classes,
    serializer=FiftyOneLabels(),
    pretrained=False,
)

# 4. Create the trainer
trainer = Trainer(fast_dev_run=True)
trainer.finetune(model, datamodule=datamodule, strategy=NoFreeze())

# 5. Finetune the model
trainer.finetune(model, datamodule=datamodule)

# 6. Save it!
trainer.save_checkpoint("video_classification.pt")

# 7. Generate predictions
predictions = trainer.predict(model, datamodule=datamodule)

# 7b. Flatten batched predictions
predictions = list(itertools.chain.from_iterable(predictions))

# 8. Add predictions to dataset and analyze
predict_dataset.set_values("flash_predictions", predictions)
session = fo.launch_app(predict_dataset)

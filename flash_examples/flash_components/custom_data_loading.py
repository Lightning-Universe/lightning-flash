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
from functools import partial
from typing import Any, Dict, List

import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.enums import LightningEnum
from torch.utils.data._utils.collate import default_collate

from flash import FlashDataset, PreprocessTransform
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.utils import download_data
from flash.core.registry import FlashRegistry

seed_everything(42)
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

#############################################################################################
#                    Use Case: Load Data from multiple folders                              #
# Imagine you have some images and they are already sorted by classes in their own folder.  #
# You would like to create your own loading mechanism that people can re-use                #
# Your loader would take a list of individual class folder and load the images from them    #
# The folder paths are independent and when loading the order of folder.                    #
# would determine the classification label.                                                 #
# Note: This is simple enough to show you the flexibility of the Flash API.                 #
#############################################################################################


#############################################################################################
#            Step 1 / 5: Create an enum to describe your new loading mechanism              #
#############################################################################################


class CustomDataTransform(LightningEnum):

    BASE = "base"
    RANDOM_ROTATION = "random_rotation"
    RANDOM_90_DEG_ROTATION = "random_90_def_rotation"


class CustomDataFormat(LightningEnum):

    MULTIPLE_FOLDERS = "multiple_folders"


#############################################################################################
#                         Step 2 / 5: Implement a FlashDataset                              #
#                                                                                           #
# A `FlashDataset` is a state-aware (c.f training, validating, testing and predicting)      #
# dataset.                                                                                  #
# and with specialized hooks (c.f load_data, load_sample) for each of those stages.         #
# The hook resolution for the function is done in the following way.                        #
# If {state}_load_data is implemented then it would be used exclusively for that stage.     #
# Otherwise, it would use the load_data function.                                           #
# If you use FlashDataset outside of Flash, the only requirements are to return a Sequence  #
# from load_data with FlashDataset or an Iterable with FlashIterableDataset.                #
# When using FlashDataset with Flash Tasks, the model expects the `load_sample` to return a #
#  dictionary with `DefaultDataKeys` as its keys (c.f `input`, `target`, metadata)          #
#                                                                                           #
#############################################################################################

FOLDER_PATH = "./data/hymenoptera_data/train"
TRAIN_FOLDERS = [os.path.join(FOLDER_PATH, "ants"), os.path.join(FOLDER_PATH, "bees")]
VAL_FOLDERS = [os.path.join(FOLDER_PATH, "ants"), os.path.join(FOLDER_PATH, "bees")]
PREDICT_FOLDER = os.path.join(FOLDER_PATH, "ants")


class MultipleFoldersImageDataset(FlashDataset):

    transforms_registry = FlashRegistry("image_classification_transform")

    def load_data(self, folders: List[str]) -> List[Dict[DefaultDataKeys, Any]]:
        if self.training:
            self.num_classes = len(folders)
        return [
            {DefaultDataKeys.INPUT: os.path.join(folder, p), DefaultDataKeys.TARGET: class_idx}
            for class_idx, folder in enumerate(folders)
            for p in os.listdir(folder)
        ]

    def load_sample(self, sample: Dict[DefaultDataKeys, Any]) -> Dict[DefaultDataKeys, Any]:
        sample[DefaultDataKeys.INPUT] = image = Image.open(sample[DefaultDataKeys.INPUT])
        sample[DefaultDataKeys.METADATA] = image.size
        return sample

    def predict_load_data(self, predict_folder: str) -> List[Dict[DefaultDataKeys, Any]]:
        assert os.path.isdir(predict_folder)
        return [{DefaultDataKeys.INPUT: os.path.join(predict_folder, p)} for p in os.listdir(predict_folder)]


train_dataset = MultipleFoldersImageDataset.from_train_data(TRAIN_FOLDERS)
val_dataset = MultipleFoldersImageDataset.from_val_data(VAL_FOLDERS)
predict_dataset = MultipleFoldersImageDataset.from_predict_data(PREDICT_FOLDER)


#############################################################################################
#                   Step 3 / 5: [optional] Implement a PreprocessTransform                       #
#                                                                                           #
# A `PreprocessTransform` is a state-aware (c.f training, validating, testing and predicting)    #
# transform. You would have to implement a `configure_transforms` hook with your transform  #
#                                                                                           #
#############################################################################################


class ImageBaseTransform(PreprocessTransform):
    def configure_per_sample_transform(self, image_size: int = 224) -> Any:
        per_sample_transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        return ApplyToKeys(DefaultDataKeys.INPUT, per_sample_transform)

    def configure_collate(self) -> Any:
        return default_collate


class ImageRandomRotationTransform(ImageBaseTransform):
    def configure_per_sample_transform(self, image_size: int = 224, rotation: float = 0) -> Any:
        transforms = [T.Resize((image_size, image_size)), T.ToTensor()]
        if self.training:
            transforms += [T.RandomRotation(rotation)]
        return ApplyToKeys(DefaultDataKeys.INPUT, T.Compose(transforms))


# Register your transform within the Flash Dataset registry
# Note: Registries can be shared by multiple dataset.
MultipleFoldersImageDataset.register_transform(CustomDataTransform.BASE, ImageBaseTransform)
MultipleFoldersImageDataset.register_transform(CustomDataTransform.RANDOM_ROTATION, ImageRandomRotationTransform)
MultipleFoldersImageDataset.register_transform(
    CustomDataTransform.RANDOM_90_DEG_ROTATION, partial(ImageRandomRotationTransform, rotation=90)
)

train_dataset = MultipleFoldersImageDataset.from_train_data(
    TRAIN_FOLDERS,
    transform=(CustomDataTransform.RANDOM_ROTATION, {"rotation": 45}),
)

print(train_dataset.transform)
# Out:
# ImageClassificationRandomRotationTransform(
#    running_stage=train,
#    transform={
#        PreprocessTransformPlacement.PER_SAMPLE_TRANSFORM: ApplyToKeys(
#           keys="input",
#           transform=Compose(
#                ToTensor()
#                RandomRotation(degrees=[-45.0, 45.0], interpolation=nearest, expand=False, fill=0)),
#        PreprocessTransformPlacement.COLLATE: default_collate,
#    },
# )

train_dataset = MultipleFoldersImageDataset.from_train_data(
    TRAIN_FOLDERS,
    transform=CustomDataTransform.RANDOM_90_DEG_ROTATION,
)

print(train_dataset.transform)
# Out:
# ImageClassificationRandomRotationTransform(
#    running_stage=train,
#    transform={
#        PreprocessTransformPlacement.PER_SAMPLE_TRANSFORM: ApplyToKeys(
#           keys="input",
#           transform=Compose(
#                ToTensor()
#                RandomRotation(degrees=[-90.0, 90.0], interpolation=nearest, expand=False, fill=0)),
#        PreprocessTransformPlacement.COLLATE: default_collate,
#    },
# )

val_dataset = MultipleFoldersImageDataset.from_val_data(VAL_FOLDERS, transform=CustomDataTransform.BASE)
print(val_dataset.transform)
# Out:
# ImageClassificationRandomRotationTransform(
#    running_stage=validate,
#    transform={
#        PreprocessTransformPlacement.PER_SAMPLE_TRANSFORM: ApplyToKeys(keys="input", transform=ToTensor()),
#        PreprocessTransformPlacement.COLLATE: default_collate,
#    },
# )

print(train_dataset[0])
# Out:
# {
#   <DefaultDataKeys.INPUT: 'input'>: <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x375 at 0x...>,
#   <DefaultDataKeys.TARGET: 'target'>: 0,
#   <DefaultDataKeys.METADATA: 'metadata'>: (500, 375)
# }

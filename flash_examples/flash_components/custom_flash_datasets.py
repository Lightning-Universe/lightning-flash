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
from contextlib import suppress
from typing import Callable, Dict, List, Optional

import torchvision.transforms as T
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.enums import LightningEnum
from torch.utils.data._utils.collate import default_collate
from torchvision.io import read_image

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.flash_dataset_container import FlashDatasetsContainer
from flash.core.data.flash_datasets import FlashDataset
from flash.core.data.flash_transform import FlashTransform, TransformPlacement
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.utils import download_data
from flash.core.registry import FlashRegistry

download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

#############################################################################################
# Imagine you have some images and they are already sorted by classes in their own folder.  #
# You would like to create your own loading mechanism that people can re-use                #
# Your loader would take a list of individual class folder and load the images from them    #
# The folder paths are independent and when loading the order of folder.                    #
# would determine the classification label.                                                 #
# Note: This is simple enough to show you the flexibility of the Flash API.                 #
#############################################################################################


#############################################################################################
#            Step 1 / 6: Create an enum to describe your new loading mechanism              #
#############################################################################################


class CustomDataTransform(LightningEnum):

    BASE = "base"
    RANDOM_ROTATION = "random_rotation"


class CustomDataFormat(LightningEnum):

    MULTIPLE_FOLDERS = "multiple_folders"


#############################################################################################
#                         Step 2 / 6: Implement a FlashDataset                              #
#                                                                                           #
# A `FlashDataset` is a state-aware (c.f training, validating, testing and predicting)      #
# dataset.                                                                                  #
# and with specialized hooks (c.f load_data, load_sample) for each of those stages.         #
# The hook resolution for the function is done in the following way.                        #
# If {state}_load_data is implemented then it would be used exclusively for that stage.     #
# Otherwise, it would use the load_data function.                                           #
#                                                                                           #
#############################################################################################

FOLDER_PATH = "./data/hymenoptera_data/train"
TRAIN_FOLDERS = [os.path.join(FOLDER_PATH, "ants"), os.path.join(FOLDER_PATH, "bees")]
VAL_FOLDERS = [os.path.join(FOLDER_PATH, "ants"), os.path.join(FOLDER_PATH, "bees")]
PREDICT_FOLDER = os.path.join(FOLDER_PATH, "ants")


class MultipleFoldersImageDataset(FlashDataset):

    transform_registry = FlashRegistry("image_classification_transform")

    def __init__(self, **dataset_kwargs):
        super().__init__()
        self.dataset_kwargs = dataset_kwargs

    def load_data(self, folders: List[str]):
        data = []
        for class_idx, folder in enumerate(folders):
            data.extend(
                [
                    {DefaultDataKeys.INPUT: os.path.join(folder, p), DefaultDataKeys.TARGET: class_idx}
                    for p in os.listdir(folder)
                ]
            )
        if self.training:
            self.num_classes = len(folders)
        return data

    def load_sample(self, sample):
        sample[DefaultDataKeys.INPUT] = image = read_image(sample[DefaultDataKeys.INPUT])
        sample[DefaultDataKeys.METADATA] = image.size
        return sample

    def predict_load_data(self, predict_folder: Optional[str]):
        return [{DefaultDataKeys.INPUT: os.path.join(predict_folder, p)} for p in os.listdir(predict_folder)]


train_dataset = MultipleFoldersImageDataset.from_data(TRAIN_FOLDERS, RunningStage.TRAINING)
val_dataset = MultipleFoldersImageDataset.from_data(VAL_FOLDERS, RunningStage.VALIDATING)
predict_dataset = MultipleFoldersImageDataset.from_data(PREDICT_FOLDER, RunningStage.PREDICTING)


#############################################################################################
#                         Step 3 / 6: Implement a FlashTransfom                             #
#                                                                                           #
# A `FlashDataset` is a state-aware (c.f training, validating, testing and predicting)      #
# dataset.                                                                                  #
# and with specialized hooks (c.f load_data, load_sample) for each of those stages.         #
# The hook resolution for the function is done in the following way.                        #
# If {state}_load_data is implemented then it would be used exclusively for that stage.     #
# Otherwise, it would use the load_data function.                                           #
#                                                                                           #
#############################################################################################


class FlashRandomRotationTransform(FlashTransform):
    def train_configure_transforms(self, rotation: float) -> Dict[TransformPlacement, Callable]:
        per_sample_transform = ApplyToKeys("input", T.Compose([T.ToTensor(), T.RandomRotation(rotation)]))
        return {
            TransformPlacement.PER_SAMPLE_TRANSFORM: per_sample_transform,
            TransformPlacement.COLLATE: default_collate,
        }

    def configure_transforms(self, **kwargs) -> Dict[TransformPlacement, Callable]:
        return {
            TransformPlacement.PER_SAMPLE_TRANSFORM: ApplyToKeys("input", T.ToTensor()),
            TransformPlacement.COLLATE: default_collate,
        }


MultipleFoldersImageDataset.register_transform(FlashRandomRotationTransform, CustomDataTransform.RANDOM_ROTATION)

train_dataset = MultipleFoldersImageDataset.from_data(
    TRAIN_FOLDERS, RunningStage.TRAINING, transform=(CustomDataTransform.RANDOM_ROTATION, {"rotation": 45})
)

print(train_dataset[0])

breakpoint()

#############################################################################################
#                             Step 3 / 5: Create a Registry                                 #
#                                                                                           #
# A registry is just a smart dictionary. Here is the key is `name` and the value `fn`.      #
# We would be registering the newly created `MultipleFoldersImage`                          #
# with the value `CustomDataFormat.MULTIPLE_FOLDERS`                                        #
#                                                                                           #
#############################################################################################


registry = FlashRegistry("image_classification_loader")
registry(fn=MultipleFoldersImageDataset, name=CustomDataFormat.MULTIPLE_FOLDERS)


#############################################################################################
#                        Step 4 / 5: Create an FlashDatasetsContainer                         #
#                                                                                           #
# The `FlashDatasetsContainer` class is a collection of DataSource which can be built out     #
# with `class_method` constructors.                                                         #
# The `FlashDatasetsContainer` requires a FlashRegistry `data_sources_registry`               #
# class attributes. By creating a `from_multiple_folders`, we can easily create a           #
# constructor taking the folders paths and by using the `cls.from_data_source`              #
# with `CustomDataFormat.MULTIPLE_FOLDERS`, it would indicate the parent class the          #
# associated `DataSource` is our newly implemented one. The extra arguments are the         #
# `{stage}_data` to be passed to the `{stage}_load_data` from the associated                #
# `MultipleFoldersImage`.                                                                   #
#                                                                                           #
#############################################################################################


class ImageClassificationContainer(FlashDatasetsContainer):

    data_sources_registry = registry
    default_data_source = CustomDataFormat.MULTIPLE_FOLDERS

    @classmethod
    def from_multiple_folders(
        cls,
        train_folders: Optional[List[str]] = None,
        val_folders: Optional[List[str]] = None,
        test_folders: Optional[List[str]] = None,
        predict_folder: Optional[str] = None,
    ):
        return cls.from_data_source(
            CustomDataFormat.MULTIPLE_FOLDERS, train_folders, val_folders, test_folders, predict_folder
        )


#############################################################################################
#                         Step 5 / 5: Finally, create the loader                            #
#############################################################################################

FOLDER_PATH = "./data/hymenoptera_data/train"

container = ImageClassificationContainer.from_multiple_folders(
    train_folders=[os.path.join(FOLDER_PATH, "ants"), os.path.join(FOLDER_PATH, "bees")],
    val_folders=[os.path.join(FOLDER_PATH, "ants"), os.path.join(FOLDER_PATH, "bees")],
    predict_folder=os.path.join(FOLDER_PATH, "ants"),
)

assert isinstance(container.train_dataset, FlashDataset)
assert isinstance(container.predict_dataset, FlashDataset)

# The ``num_classes`` value was set line 76.
assert container.train_dataset.num_classes == 2

# The ``num_classes`` value was set only for training as `self.training` was used,
# so it doesn't exist for the predict_dataset
with suppress(AttributeError):
    container.val_dataset.num_classes

# As test_data weren't provided, the test dataset is None.
assert not container.test_dataset


print(container.train_dataset[0])
# out:
# {
#   <DefaultDataKeys.INPUT: 'input'>: 'data/hymenoptera_data/train/ants/957233405_25c1d1187b.jpg',
#   <DefaultDataKeys.TARGET: 'target'>: 0
# }

assert isinstance(container.predict_dataset, FlashDataset)
print(container.predict_dataset[0])
# out:
# {
#   {<DefaultDataKeys.INPUT: 'input'>: 'data/hymenoptera_data/train/ants/957233405_25c1d1187b.jpg'}
# }

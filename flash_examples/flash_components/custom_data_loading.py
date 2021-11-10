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
from functools import partial
from typing import Any, Dict, List, Optional

import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data._utils.collate import default_collate

from flash import _PACKAGE_ROOT, FlashDataset
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.data.new_data_module import DataModule
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.utils import download_data

seed_everything(42)
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", f"{_PACKAGE_ROOT}/data")

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
#                         Step 1 / 2: Implement a FlashDataset                              #
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

FOLDER_PATH = f"{_PACKAGE_ROOT}/data/hymenoptera_data/train"
TRAIN_FOLDERS = [os.path.join(FOLDER_PATH, "ants"), os.path.join(FOLDER_PATH, "bees")]
VAL_FOLDERS = [os.path.join(FOLDER_PATH, "ants"), os.path.join(FOLDER_PATH, "bees")]
PREDICT_FOLDER = os.path.join(FOLDER_PATH, "ants")


class MultipleFoldersImageDataset(FlashDataset):
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
#                   Step 2 / 2: [optional] Implement a InputTransform                       #
#                                                                                           #
# A `InputTransform` is a state-aware (c.f training, validating, testing and predicting)    #
# transform. You would have to implement a `configure_transforms` hook with your transform  #
#                                                                                           #
#############################################################################################


class BaseImageInputTransform(InputTransform):
    def configure_per_sample_transform(self, image_size: int = 224) -> Any:
        per_sample_transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        return ApplyToKeys(DefaultDataKeys.INPUT, per_sample_transform)

    def configure_collate(self) -> Any:
        return default_collate


class ImageRandomRotationInputTransform(BaseImageInputTransform):
    def configure_per_sample_transform(self, image_size: int = 224, rotation: float = 0) -> Any:
        transforms = [T.Resize((image_size, image_size)), T.ToTensor()]
        if self.training:
            transforms += [T.RandomRotation(rotation)]
        return ApplyToKeys(DefaultDataKeys.INPUT, T.Compose(transforms))


# Register your transform within the Flash Dataset registry
# Note: Registries can be shared by multiple dataset.
MultipleFoldersImageDataset.register_input_transform("base", BaseImageInputTransform)
MultipleFoldersImageDataset.register_input_transform("random_rotation", ImageRandomRotationInputTransform)
MultipleFoldersImageDataset.register_input_transform(
    "random_90_def_rotation", partial(ImageRandomRotationInputTransform, rotation=90)
)

train_dataset = MultipleFoldersImageDataset.from_train_data(
    TRAIN_FOLDERS,
    transform=("random_rotation", {"rotation": 45}),
)

print(train_dataset.transform)
# Out:
# ImageClassificationRandomRotationTransform(
#    running_stage=train,
#    transform={
#        InputTransformPlacement.PER_SAMPLE_TRANSFORM: ApplyToKeys(
#           keys="input",
#           transform=Compose(
#                ToTensor()
#                RandomRotation(degrees=[-45.0, 45.0], interpolation=nearest, expand=False, fill=0)),
#        InputTransformPlacement.COLLATE: default_collate,
#    },
# )

train_dataset = MultipleFoldersImageDataset.from_train_data(
    TRAIN_FOLDERS,
    transform="random_90_def_rotation",
)

print(train_dataset.transform)
# Out:
# ImageClassificationRandomRotationTransform(
#    running_stage=train,
#    transform={
#        InputTransformPlacement.PER_SAMPLE_TRANSFORM: ApplyToKeys(
#           keys="input",
#           transform=Compose(
#                ToTensor()
#                RandomRotation(degrees=[-90.0, 90.0], interpolation=nearest, expand=False, fill=0)),
#        InputTransformPlacement.COLLATE: default_collate,
#    },
# )

val_dataset = MultipleFoldersImageDataset.from_val_data(VAL_FOLDERS, transform="base")
print(val_dataset.transform)
# Out:
# ImageClassificationRandomRotationTransform(
#    running_stage=validate,
#    transform={
#        InputTransformPlacement.PER_SAMPLE_TRANSFORM: ApplyToKeys(keys="input", transform=ToTensor()),
#        InputTransformPlacement.COLLATE: default_collate,
#    },
# )

print(train_dataset[0])
# Out:
# {
#   <DefaultDataKeys.INPUT: 'input'>: <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x375 at 0x...>,
#   <DefaultDataKeys.TARGET: 'target'>: 0,
#   <DefaultDataKeys.METADATA: 'metadata'>: (500, 375)
# }

#############################################################################################
#                           Step 4 / 5: Create a DataModule                                 #
#                                                                                           #
# The `DataModule` class is a collection of FlashDataset and you can pass them directly to  #
# its init function.                                                                        #
#                                                                                           #
#############################################################################################


datamodule = DataModule(
    train_dataset=MultipleFoldersImageDataset.from_train_data(TRAIN_FOLDERS, transform="random_rotation"),
    val_dataset=MultipleFoldersImageDataset.from_val_data(VAL_FOLDERS, transform="base"),
    predict_dataset=MultipleFoldersImageDataset.from_predict_data(PREDICT_FOLDER, transform="base"),
    batch_size=2,
)


assert isinstance(datamodule.train_dataset, FlashDataset)
assert isinstance(datamodule.predict_dataset, FlashDataset)

# The ``num_classes`` value was set line 89.
assert datamodule.train_dataset.num_classes == 2

# The ``num_classes`` value was set only for training as `self.training` was used,
# so it doesn't exist for the predict_dataset
with suppress(AttributeError):
    datamodule.val_dataset.num_classes

# As test_data weren't provided, the test dataset is None.
assert not datamodule.test_dataset


print(datamodule.train_dataset[0])
# Out:
# {
#   <DefaultDataKeys.INPUT: 'input'>: <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x375 at 0x...>,
#   <DefaultDataKeys.TARGET: 'target'>: 0,
#   <DefaultDataKeys.METADATA: 'metadata'>: (500, 375)
# }

assert isinstance(datamodule.predict_dataset, FlashDataset)
print(datamodule.predict_dataset[0])
# out:
# {
#   {<DefaultDataKeys.INPUT: 'input'>: 'data/hymenoptera_data/train/ants/957233405_25c1d1187b.jpg'}
# }


# access the dataloader, the collate_fn will be injected directly within the dataloader from the provided transform
batch = next(iter(datamodule.train_dataloader()))
# Out:
# {
#   <DefaultDataKeys.INPUT: 'input'>: tensor([...]),
#   <DefaultDataKeys.TARGET: 'target'>: tensor([...]),
#   <DefaultDataKeys.METADATA: 'metadata'>: [(...), (...), ...],
# }
print(batch)


#############################################################################################
#                Step 5 / 5: Provide your new utility with your DataModule                  #
#                                                                                           #
# The `DataModule` class is a collection of FlashDataset and you can pass them directly to  #
# its init function.                                                                        #
#                                                                                           #
#############################################################################################


class ImageClassificationDataModule(DataModule):
    @classmethod
    def from_multiple_folders(
        cls,
        train_folders: Optional[List[str]] = None,
        val_folders: Optional[List[str]] = None,
        test_folders: Optional[List[str]] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[INPUT_TRANSFORM_TYPE] = None,
        val_transform: Optional[INPUT_TRANSFORM_TYPE] = None,
        test_transform: Optional[INPUT_TRANSFORM_TYPE] = None,
        predict_transform: Optional[INPUT_TRANSFORM_TYPE] = None,
        **data_module_kwargs: Any,
    ) -> "ImageClassificationDataModule":

        return cls(
            *cls.create_flash_datasets(
                "multiple_folders",
                train_folders,
                val_folders,
                test_folders,
                predict_folder,
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
            ),
            **data_module_kwargs,
        )


ImageClassificationDataModule.register_flash_dataset("multiple_folders", MultipleFoldersImageDataset)


# Create the datamodule with your new constructor. This is purely equivalent to the previous datamdoule creation.
datamodule = ImageClassificationDataModule.from_multiple_folders(
    train_folders=TRAIN_FOLDERS,
    val_folders=VAL_FOLDERS,
    predict_folder=PREDICT_FOLDER,
    train_transform="random_rotation",
    val_transform="base",
    predict_transform="base",
    batch_size=2,
)

# access the dataloader, the collate_fn will be injected directly within the dataloader from the provided transform
batch = next(iter(datamodule.train_dataloader()))
# Out:
# {
#   <DefaultDataKeys.INPUT: 'input'>: tensor([...]),
#   <DefaultDataKeys.TARGET: 'target'>: tensor([...]),
#   <DefaultDataKeys.METADATA: 'metadata'>: [(...), (...), ...],
# }
print(batch)

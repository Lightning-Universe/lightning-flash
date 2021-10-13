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
from typing import Any, Dict, List

import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.enums import LightningEnum
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from flash import _PACKAGE_ROOT
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.datapipes import FlashDataPipes
from flash.core.data.preprocess_transform import PreprocessTransform
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.utils import download_data

seed_everything(42)
ROOT_DATA = f"{_PACKAGE_ROOT}/data"
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", ROOT_DATA)

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

FOLDER_PATH = f"{ROOT_DATA}/hymenoptera_data/train"
TRAIN_FOLDERS = [os.path.join(FOLDER_PATH, "ants"), os.path.join(FOLDER_PATH, "bees")]
VAL_FOLDERS = [os.path.join(FOLDER_PATH, "ants"), os.path.join(FOLDER_PATH, "bees")]
PREDICT_FOLDER = os.path.join(FOLDER_PATH, "ants")


class ListImagesFromFolders(FlashDataPipes):
    def __init__(self, folders: List[str], running_stage: RunningStage):
        super().__init__(running_stage=running_stage)
        self.folders = folders

    def __len__(self) -> int:
        if isinstance(self.folders, str):
            self.folders = [self.folders]
        return sum(len(os.listdir(f)) for f in self.folders)

    def process_data(self) -> List[Dict[DefaultDataKeys, Any]]:
        if self.training:
            self.num_classes = len(self.folders)
        return [
            {DefaultDataKeys.INPUT: os.path.join(folder, p), DefaultDataKeys.TARGET: class_idx}
            for class_idx, folder in enumerate(self.folders)
            for p in os.listdir(folder)
        ]

    def predict_process_data(self) -> List[Dict[DefaultDataKeys, Any]]:
        assert os.path.isdir(self.folders)
        return [{DefaultDataKeys.INPUT: os.path.join(self.folders, p)} for p in os.listdir(self.folders)]


class LoadImage(FlashDataPipes):
    def __init__(self, data_pipe: FlashDataPipes, running_stage: RunningStage):
        super().__init__(running_stage=running_stage)
        self.data_pipe = data_pipe

    def process_data(self, sample: Dict[DefaultDataKeys, Any]) -> Dict[DefaultDataKeys, Any]:
        sample[DefaultDataKeys.INPUT] = image = Image.open(sample[DefaultDataKeys.INPUT])
        sample[DefaultDataKeys.METADATA] = image.size
        return sample


class ImageBaseTransform(PreprocessTransform):
    def configure_per_sample_transform(self, image_size: int = 224, rotation: float = 0) -> Any:
        transforms = [T.Resize((image_size, image_size)), T.ToTensor()]
        if self.training:
            transforms += [T.RandomRotation(rotation)]
        return ApplyToKeys(DefaultDataKeys.INPUT, T.Compose(transforms))

    def configure_collate(self) -> Any:
        return default_collate


class PreprocessTransformDataPipes(FlashDataPipes):
    def __init__(
        self, data_pipe: LoadImage, batch_size: int, transform: PreprocessTransform, running_stage: RunningStage
    ):
        super().__init__(running_stage=running_stage)

        self.data_pipe = data_pipe.map(lambda x: self.transform.per_sample_transform(x)).batch(batch_size)
        self.transform = transform

    def process_data(self, list_of_samples: Any):
        batch = self.transform.collate(list_of_samples)
        return self.transform.per_batch_transform(batch)


train_data_pipe = ListImagesFromFolders.from_train_data(TRAIN_FOLDERS)
train_data_pipe = LoadImage.from_train_data(train_data_pipe)
train_data_pipe = PreprocessTransformDataPipes.from_train_data(
    train_data_pipe, batch_size=2, transform=ImageBaseTransform(running_stage=RunningStage.TRAINING)
)
dataloader = DataLoader(train_data_pipe, collate_fn=lambda x: x)
batch = next(iter(dataloader))
breakpoint()
# Out:
# {
#   <DefaultDataKeys.INPUT: 'input'>: tensor([[[[...]]]]),
#   <DefaultDataKeys.TARGET: 'target'>: tensor([0, 0]),
#   <DefaultDataKeys.METADATA: 'metadata'>: tensor([500, 500]), tensor([375, 500])]
# }

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
from typing import List, Optional

from pytorch_lightning.utilities.enums import LightningEnum

from flash.core.data.auto_dataset import AutoDataset
from flash.core.data.auto_dataset_container import AutoDatasetContainer
from flash.core.data.data_source import DataSource, DefaultDataKeys
from flash.core.data.utils import download_data
from flash.core.registry import FlashRegistry

download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

"""
The following code implements the following use-case from scratch.

Imagine you have some images and they are already sorted by classes in their own folder.
You would like to create your own loading mechanism people can re-use
where people can load only a list of classes they are interested in.
Importantly, the folders can be independent and not located at the same place.
Note: This is simple enough to show you the flexibility of the Flash API.
"""

# Step 1: Create an enum to describe your new loading mechanism


class CustomDataFormat(LightningEnum):

    MULTIPLE_FOLDERS = "multiple_folders"


"""
Step 2: Implement a DataSource.
A DataSource is a state-aware (c.f training, validating, testing and predicting) dataset
and with specialized hooks (c.f load_data, load_sample) for each of those stages.
The hook resolution for the function is done in the following way.
If {state}_load_data is implemented then it would be used exclusively for that stage.
Otherwise, it would use the load_data function.
`DataSource` can transform themselves into their PyTorch IterableDataset / Dataset counterpart
which is an AutoDataset/AutoIterableDataset in Flash.
"""


class MultipleFoldersImage(DataSource):
    def __init__(self, **data_source_kwargs):
        super().__init__()
        self.data_source_kwargs = data_source_kwargs

    def load_data(self, folders: Optional[List[str]], dataset: AutoDataset):
        data = []
        for class_idx, folder in enumerate(folders):
            data.extend(
                [
                    {DefaultDataKeys.INPUT: os.path.join(folder, p), DefaultDataKeys.TARGET: class_idx}
                    for p in os.listdir(folder)
                ]
            )
        if self.training:
            dataset.num_classes = len(folders)
        return data

    def load_sample(self, sample):
        return sample

    def predict_load_data(self, predict_folder: Optional[str]):
        return [{DefaultDataKeys.INPUT: os.path.join(predict_folder, p)} for p in os.listdir(predict_folder)]


"""
Step 3: Create a Registry
A registry is just a smart dictionary. Here is the key is `name` and the value `fn`.
We would be registering the newly created `MultipleFoldersImage`
with the value `CustomDataFormat.MULTIPLE_FOLDERS`
"""

registry = FlashRegistry("image_classification_loader")
registry(fn=MultipleFoldersImage, name=CustomDataFormat.MULTIPLE_FOLDERS)

"""
Step 4: Create an AutoDatasetContainer
The `AutoDatasetContainer` class is a collection of DataSource which can be built out
with `class_method` constructors.
The `AutoDatasetContainer` requires a FlashRegistry `data_sources_registry` class attributes.
By creating a `from_multiple_folders`, we can easily create a constructor taking the folders paths
and by using the `cls.from_data_source` with `CustomDataFormat.MULTIPLE_FOLDERS`,
it would indicate the parent class the associated `DataSource` is our newly implemented one.
The extra arguments are the `{stage}_data` to be passed to the `{stage}_load_data`
from the associated `MultipleFoldersImage`
"""


class ImageClassificationLoader(AutoDatasetContainer):

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


"""
Step 5: Finally, create our loader.
"""
FOLDER_PATH = "./data/hymenoptera_data/train"

loader = ImageClassificationLoader.from_multiple_folders(
    train_folders=[os.path.join(FOLDER_PATH, "ants"), os.path.join(FOLDER_PATH, "bees")],
    predict_folder=os.path.join(FOLDER_PATH, "ants"),
)

assert isinstance(loader.train_dataset, AutoDataset)
# the ``num_classes`` value was set line 64.
assert loader.train_dataset.num_classes == 2
print(loader.train_dataset[0])
# out:
# {
#   <DefaultDataKeys.INPUT: 'input'>: 'data/hymenoptera_data/train/ants/957233405_25c1d1187b.jpg',
#   <DefaultDataKeys.TARGET: 'target'>: 0
# }

assert isinstance(loader.predict_dataset, AutoDataset)
print(loader.predict_dataset[0])
# out:
# {
#   {<DefaultDataKeys.INPUT: 'input'>: 'data/hymenoptera_data/train/ants/957233405_25c1d1187b.jpg'}
# }

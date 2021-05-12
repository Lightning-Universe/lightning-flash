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
import torchvision
from torch import nn

from flash.core.classification import Labels
from flash.core.finetuning import FreezeUnfreeze
from flash.data.utils import download_data
from flash.utils.cli import FlashCLI
from flash.vision import ImageClassificationData, ImageClassifier

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")


# 2.a Optional: Register a custom backbone
# This is useful to create new backbone and make them accessible from `ImageClassifier`
@ImageClassifier.backbones(name="resnet18")
def fn_resnet(pretrained: bool = True):
    model = torchvision.models.resnet18(pretrained)
    # remove the last two layers & turn it into a Sequential model
    backbone = nn.Sequential(*list(model.children())[:-2])
    num_features = model.fc.in_features
    # backbones need to return the num_features to build the head
    return backbone, num_features


# 2.b Optional: List available backbones
print(ImageClassifier.available_backbones())


class ImageClassificationCLI(FlashCLI):

    def add_arguments_to_parser(self, parser):
        parser.set_defaults({
            'data.train_folder': 'data/hymenoptera_data/train/',
            'data.val_folder': 'data/hymenoptera_data/val/',
            'data.test_folder': 'data/hymenoptera_data/test/',
        })

    def parse_arguments(self):
        # ignore the fact that this is needed - might be a bug
        self.config = self.parser.parse_args(_skip_check=True)

    def instantiate_datamodule(self):
        # Link the num_classes for the model
        self.config_init["model"]["num_classes"] = self.datamodule.num_classes

    def prepare_fit_kwargs(self):
        super().prepare_fit_kwargs()
        # TODO: expose the strategy arguments?
        self.fit_kwargs["strategy"] = FreezeUnfreeze(unfreeze_epoch=1)

    def fit(self):
        self.trainer.finetune(**self.fit_kwargs)


# 3. Build the model, datamodule, and trainer. Expose them through CLI. Fine-tune
cli = ImageClassificationCLI(ImageClassifier, ImageClassificationData, datasource="paths")

# 4. Save a checkpoint!
cli.trainer.save_checkpoint("image_classification_model.pt")

# 5a. Predict what's on a few images! ants or bees?
# Serialize predictions as labels, automatically inferred from the training data in part 2.
cli.model.serializer = Labels()
predictions = cli.model.predict([
    "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
    "data/hymenoptera_data/val/bees/590318879_68cf112861.jpg",
    "data/hymenoptera_data/val/ants/540543309_ddbb193ee5.jpg",
])
print(predictions)

# 5b. Or generate predictions with a whole folder!
datamodule = ImageClassificationData.from_folders(predict_folder="data/hymenoptera_data/predict/")
predictions = cli.trainer.predict(cli.model, datamodule=datamodule)
print(predictions)

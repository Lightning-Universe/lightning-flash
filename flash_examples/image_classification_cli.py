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
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier
from flash.utils.cli import FlashCLI

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")


class ImageClassificationCLI(FlashCLI):

    # def add_arguments_to_parser(self, parser):
    #     parser.set_defaults({
    #         'data.folders.train_folder': 'data/hymenoptera_data/train/',
    #         'data.folders.val_folder': 'data/hymenoptera_data/val/',
    #         'data.folders.test_folder': 'data/hymenoptera_data/test/',
    #     })

    def prepare_fit_kwargs(self):
        super().prepare_fit_kwargs()
        # TODO: expose the strategy arguments?
        self.fit_kwargs["strategy"] = "freeze"

    def fit(self) -> None:
        """Runs fit of the instantiated trainer class and prepared fit keyword arguments"""
        self.trainer.finetune(**self.fit_kwargs)


# 2. Build the model, datamodule, and trainer. Expose them through CLI. Fine-tune
cli = ImageClassificationCLI(ImageClassifier, ImageClassificationData)

# 3. Predict what's on a few images! ants or bees?
predictions = cli.model.predict([
    "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
    "data/hymenoptera_data/val/bees/590318879_68cf112861.jpg",
    "data/hymenoptera_data/val/ants/540543309_ddbb193ee5.jpg",
])
print(predictions)

# 4. Save the model!
cli.trainer.save_checkpoint("image_classification_model.pt")

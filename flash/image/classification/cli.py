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
from enum import Enum
from typing import Optional

from flash.core.data.utils import download_data
from flash.core.utilities.flash_cli import FlashCLI
from flash.core.utilities.lightning_cli import LightningArgumentParser
from flash.image import ImageClassificationData, ImageClassifier


class _DemoOptions(str, Enum):
    movies = "movies"
    ants_and_bees = "ants_and_bees"


class _AntsAndBeesCLI(FlashCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.set_defaults(
            {
                "data.train_folder": "data/hymenoptera_data/train/",
                "data.val_folder": "data/hymenoptera_data/val/",
                "batch_size": 4,
                "num_workers": 0,
            }
        )

    def before_instantiate_classes(self) -> None:
        download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")


class _MoviesCLI(FlashCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.set_defaults(
            {
                "data.input_field": "Id",
                "data.target_fields": ["Action", "Romance", "Crime", "Thriller", "Adventure"],
                "data.train_file": "data/movie_posters/train/metadata.csv",
                "data.val_file": "data/movie_posters/val/metadata.csv",
                "batch_size": 4,
                "num_workers": 0,
            }
        )

    def before_instantiate_classes(self) -> None:
        download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip", "./data")


def cli(demo: Optional[_DemoOptions] = None) -> None:
    """Classify images.

    Args:
        demo: Which demo to use, or none.
    """
    if demo is None:
        pass  # cli = FlashCLI(
        #    ImageClassifier,
        #    ImageClassificationData,
        #    default_datamodule_builder=from_hymenoptera,
        #    additional_datamodule_builders=[from_movie_posters],
        #    default_arguments={"trainer.max_epochs": 3},
        #    datamodule_attributes={"num_classes", "multi_label"},
        #    legacy=True,
        # )
        # cli.trainer.save_checkpoint("image_classification_model.pt")
    elif demo == _DemoOptions.ants_and_bees:
        _AntsAndBeesCLI(ImageClassifier, ImageClassificationData.from_folders)
    elif demo == _DemoOptions.movies:
        _MoviesCLI(ImageClassifier, ImageClassificationData.from_csv)


if __name__ == "__main__":
    cli()

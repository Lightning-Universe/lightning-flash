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
import sys

sys.path
sys.path.append('../../../flash')

from flash.core.utilities.flash_cli import FlashCLI  # noqa: E402
from flash.image import ImageClassificationData, ImageClassifier  # noqa: E402

# 1. Build the model, datamodule, and trainer. Expose them through CLI. Fine-tune
cli = FlashCLI(
    ImageClassifier,
    ImageClassificationData,
    default_subcommand="from_folders",
    default_arguments={
        'trainer.max_epochs': 3,
        'model.backbone': 'resnet18',
        'from_folders.train_folder': 'data/hymenoptera_data/train/',
        'from_folders.val_folder': 'data/hymenoptera_data/val/',
        'from_folders.test_folder': 'data/hymenoptera_data/test/',
    }
)

# 2. Save the model!
cli.trainer.save_checkpoint("image_classification_model.pt")

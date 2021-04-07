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

import flash
from flash import Trainer
from flash.core.finetuning import FreezeUnfreeze
from flash.data.utils import download_data
from flash.vision import IMAGE_CLASSIFIER_BACKBONES, ImageClassificationData, ImageClassifier

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

# 2. Load the data
datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
    test_folder="data/hymenoptera_data/test/",
)


@ImageClassifier.register_function(name="username/resnet18")
def fn_resnet(pretrained: bool = True):
    model = getattr(torchvision.models, "resnet18", None)(pretrained)
    backbone = nn.Sequential(*list(model.children())[:-2])
    num_features = model.fc.in_features
    # backbones need to return the num_features to build the head
    return backbone, num_features


# 3. Build the model
model = ImageClassifier(backbone="username/resnet18", num_classes=datamodule.num_classes)

# 4. Create the trainer.
trainer = flash.Trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=1)

# 5. Train the model
trainer.finetune(model, datamodule=datamodule, strategy=FreezeUnfreeze(unfreeze_epoch=1))

# 3a. Predict what's on a few images! ants or bees?
predictions = model.predict([
    "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
    "data/hymenoptera_data/val/bees/590318879_68cf112861.jpg",
    "data/hymenoptera_data/val/ants/540543309_ddbb193ee5.jpg",
])

print(predictions)

datamodule = ImageClassificationData.from_folders(predict_folder="data/hymenoptera_data/predict/")

# 3b. Or generate predictions with a whole folder!
predictions = Trainer().predict(model, datamodule=datamodule)
print(predictions)

# 4. Save it!
trainer.save_checkpoint("image_classification_model.pt")

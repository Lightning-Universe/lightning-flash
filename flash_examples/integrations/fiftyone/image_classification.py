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
from itertools import chain

import torch

import flash
from flash.core.classification import FiftyOneLabels, Labels
from flash.core.data.utils import download_data
from flash.core.finetuning import FreezeUnfreeze
from flash.core.integrations.fiftyone import visualize
from flash.image import ImageClassificationData, ImageClassifier

# 1 Download data
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip")

# 2 Load data
datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
    test_folder="data/hymenoptera_data/test/",
    predict_folder="data/hymenoptera_data/predict/",
)

# 3 Fine tune a model
model = ImageClassifier(
    backbone="resnet18",
    num_classes=datamodule.num_classes,
    output=Labels(),
)
trainer = flash.Trainer(
    max_epochs=1,
    gpus=torch.cuda.device_count(),
    limit_train_batches=1,
    limit_val_batches=1,
)
trainer.finetune(
    model,
    datamodule=datamodule,
    strategy=FreezeUnfreeze(unfreeze_epoch=1),
)
trainer.save_checkpoint("image_classification_model.pt")

# 4 Predict from checkpoint
model = ImageClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/image_classification_model.pt")
model.output = FiftyOneLabels(return_filepath=True)  # output FiftyOne format
predictions = trainer.predict(model, datamodule=datamodule)
predictions = list(chain.from_iterable(predictions))  # flatten batches

# 5 Visualize predictions in FiftyOne App
# Optional: pass `wait=True` to block execution until App is closed
session = visualize(predictions)

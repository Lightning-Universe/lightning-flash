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
import torch

import flash
from flash.core.data.utils import download_data
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
    batch_size=16,
)

# 3 Fine tune a model
model = ImageClassifier(
    backbone="resnet18",
    labels=datamodule.labels,
)
trainer = flash.Trainer(
    max_epochs=1,
    gpus=torch.cuda.device_count(),
    fast_dev_run=True,
)
trainer.finetune(
    model,
    datamodule=datamodule,
    strategy=("freeze_unfreeze", 1),
)
trainer.save_checkpoint("image_classification_model.pt")

# 4 Predict from checkpoint
model = ImageClassifier.load_from_checkpoint("image_classification_model.pt")
predictions = trainer.predict(model, datamodule=datamodule, output="fiftyone")  # output FiftyOne format

# 5 Visualize predictions in FiftyOne App
# Optional: pass `wait=True` to block execution until App is closed
session = visualize(predictions, wait=True)

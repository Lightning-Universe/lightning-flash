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
from flash.image import SemanticSegmentation, SemanticSegmentationData

# 1. Create the DataModule
# The data was generated with the  CARLA self-driving simulator as part of the Kaggle Lyft Udacity Challenge.
# More info here: https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge
download_data(
    "https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip",
    "./data",
)

datamodule = SemanticSegmentationData.from_folders(
    train_folder="data/CameraRGB",
    train_target_folder="data/CameraSeg",
    val_split=0.1,
    image_size=(256, 256),
    num_classes=21,
)

# 2. Build the task
model = SemanticSegmentation(
    backbone="mobilenetv3_large_100",
    head="fpn",
    num_classes=datamodule.num_classes,
)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count(), fast_dev_run=True)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Segment a few images!
predict_datamodule = SemanticSegmentationData.from_files(
    predict_files=[
        "data/CameraRGB/F61-1.png",
        "data/CameraRGB/F62-1.png",
        "data/CameraRGB/F63-1.png",
    ]
)
predictions = trainer.predict(model, datamodule=predict_datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("semantic_segmentation_model.pt")

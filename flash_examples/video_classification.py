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
from flash.video import VideoClassificationData, VideoClassifier

# 1. Create the DataModule
# Find more datasets at https://pytorchvideo.readthedocs.io/en/latest/data.html
download_data("https://pl-flash-data.s3.amazonaws.com/kinetics.zip", "./data")

datamodule = VideoClassificationData.from_folders(
    train_folder="data/kinetics/train",
    val_folder="data/kinetics/val",
    clip_sampler="uniform",
    clip_duration=1,
    decode_audio=False,
    batch_size=1,
)

# 2. Build the task
model = VideoClassifier(backbone="x3d_xs", labels=datamodule.labels, pretrained=False)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(
    max_epochs=1, gpus=torch.cuda.device_count(), strategy="ddp" if torch.cuda.device_count() > 1 else None
)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Make a prediction
datamodule = VideoClassificationData.from_folders(predict_folder="data/kinetics/predict", batch_size=1)
predictions = trainer.predict(model, datamodule=datamodule, output="labels")
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("video_classification.pt")

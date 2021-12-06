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
from flash.audio import AudioClassificationData
from flash.core.data.utils import download_data
from flash.image import ImageClassifier

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/urban8k_images.zip", "./data")

datamodule = AudioClassificationData.from_folders(
    train_folder="data/urban8k_images/train",
    val_folder="data/urban8k_images/val",
    spectrogram_size=(64, 64),
)

# 2. Build the model.
model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy=("freeze_unfreeze", 1))

# 4. Predict what's on few images! air_conditioner, children_playing, siren e.t.c
predictions = AudioClassificationData.from_files(
    predict_files=[
        "data/urban8k_images/test/air_conditioner/13230-0-0-5.wav.jpg",
        "data/urban8k_images/test/children_playing/9223-2-0-15.wav.jpg",
        "data/urban8k_images/test/jackhammer/22883-7-10-0.wav.jpg",
    ]
)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("audio_classification_model.pt")

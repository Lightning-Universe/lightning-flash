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
import flash
from flash.audio import AudioSourceSeparationData, AudioSourceSeparator
from flash.core.data.utils import download_data

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")  # To-Do : Add a right path

datamodule = AudioSourceSeparationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
)

# 2. Build the task
model = AudioSourceSeparator(backbone="convtasnet", n_src=datamodule.n_src)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=2)
# trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Predict what's on a few audios! ants or bees?
predictions = model.predict(
    [  # To-Do : check if model.predict can have both list and string.
        "MiniLibriMix/val/mix_clean/100-121669-0026_718-129597-0003.wav",
        "MiniLibriMix/val/mix_clean/1025-92820-0032_8410-278217-0015.wav",
        "MiniLibriMix/val/mix_clean/1027-125140-0032_8388-275212-0023.wav",
    ]
)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("audio_source_seperation_model.pt")

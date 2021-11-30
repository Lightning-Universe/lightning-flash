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
from flash.text import TranslationData, TranslationTask

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/wmt_en_ro.zip", "./data")

datamodule = TranslationData.from_csv(
    "input",
    "target",
    train_file="data/wmt_en_ro/train.csv",
    val_file="data/wmt_en_ro/valid.csv",
    backbone="Helsinki-NLP/opus-mt-en-ro",
)

# 2. Build the task
model = TranslationTask(backbone="Helsinki-NLP/opus-mt-en-ro")

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Translate something!
predictions = model.predict(
    [
        "BBC News went to meet one of the project's first graduates.",
        "A recession has come as quickly as 11 months after the first rate hike and as long as 86 months.",
        "Of course, it's still early in the election cycle.",
    ]
)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("translation_model_en_ro.pt")

import pytorch_lightning as pl

pl.seed_everything(42)
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

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/wmt_en_ro.zip", "data/")

backbone = "Helsinki-NLP/opus-mt-en-ro"

# 2. Load the data
datamodule = TranslationData.from_csv(
    "input",
    "target",
    train_file="data/wmt_en_ro/train.csv",
    val_file="data/wmt_en_ro/valid.csv",
    test_file="data/wmt_en_ro/test.csv",
    batch_size=1,
    backbone=backbone,
)

# 3. Build the model
model = TranslationTask(backbone=backbone)

# 4. Create the trainer
trainer = flash.Trainer(
    precision=16 if torch.cuda.is_available() else 32,
    gpus=int(torch.cuda.is_available()),
    fast_dev_run=True,
)

# 5. Fine-tune the model
trainer.finetune(model, datamodule=datamodule)

# 6. Test model
trainer.test(model)

# 7. Save it!
trainer.save_checkpoint("translation_model_en_ro.pt")

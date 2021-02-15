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
from flash import download_data
from flash.text import TranslationData, TranslationTask

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/wmt_en_ro.zip", "data/")

# 2. Load the data
datamodule = TranslationData.from_files(
    train_file="data/wmt_en_ro/train.csv",
    valid_file="data/wmt_en_ro/valid.csv",
    test_file="data/wmt_en_ro/test.csv",
    input="input",
    target="target",
)

# 3. Build the model
model = TranslationTask()

# 4. Create the trainer. Run once on data
trainer = flash.Trainer(max_epochs=1, precision=16, gpus=1)

# 5. Fine-tune the model
trainer.finetune(model, datamodule=datamodule)

# 6. Test model
trainer.test()

# 7. Save it!
trainer.save_checkpoint("translation_model_en_ro.pt")

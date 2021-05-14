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
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "data/")

# 2. Load the data
datamodule = TextClassificationData.from_csv(
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv",
    input_fields="review",
    target_fields="sentiment",
    batch_size=16,
)

# 3. Build the model
model = TextClassifier(num_classes=datamodule.num_classes)

# 4. Create the trainer
trainer = flash.Trainer(fast_dev_run=True)

# 5. Fine-tune the model
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 6. Test model
trainer.test(model)

# 7. Save it!
trainer.save_checkpoint("text_classification_model.pt")

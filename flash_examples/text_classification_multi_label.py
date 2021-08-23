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
from flash.text import TextClassificationData, TextClassifier

# 1. Create the DataModule
# Data from the Kaggle Toxic Comment Classification Challenge:
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
download_data("https://pl-flash-data.s3.amazonaws.com/jigsaw_toxic_comments.zip", "./data")

datamodule = TextClassificationData.from_csv(
    "comment_text",
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
    train_file="data/jigsaw_toxic_comments/train.csv",
    val_split=0.1,
    backbone="unitary/toxic-bert",
)

# 2. Build the task
model = TextClassifier(
    backbone="unitary/toxic-bert",
    num_classes=datamodule.num_classes,
    multi_label=True,
)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Generate predictions for a few comments!
predictions = model.predict(
    [
        "No, he is an arrogant, self serving, immature idiot. Get it right.",
        "U SUCK HANNAH MONTANA",
        "Would you care to vote? Thx.",
    ]
)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("text_classification_multi_label_model.pt")

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
from flash.tabular import TabularClassificationData, TabularClassifier

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", "./data")

datamodule = TabularClassificationData.from_csv(
    ["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
    "Fare",
    target_fields="Survived",
    train_file="data/titanic/titanic.csv",
    val_split=0.1,
)

# 2. Build the task
model = TabularClassifier.from_data(datamodule)

# 3. Create the trainer and train the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.fit(model, datamodule=datamodule)

# 4. Generate predictions from a CSV
predict_datamodule = TabularClassificationData.from_csv(predict_file="data/titanic/titanic.csv")
predictions = trainer.predict(model, datamodule=predict_datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("tabular_classification_model.pt")

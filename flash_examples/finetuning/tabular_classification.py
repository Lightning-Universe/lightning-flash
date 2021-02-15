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
from pytorch_lightning.metrics.classification import Accuracy, Precision, Recall

import flash
from flash.core.data import download_data
from flash.tabular import TabularClassifier, TabularData

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", "data/")

# 2. Load the data
datamodule = TabularData.from_csv(
    "./data/titanic/titanic.csv",
    test_csv="./data/titanic/test.csv",
    categorical_input=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
    numerical_input=["Fare"],
    target="Survived",
    val_size=0.25,
)

# 3. Build the model
model = TabularClassifier.from_data(datamodule, metrics=[Accuracy(), Precision(), Recall()])

# 4. Create the trainer. Run 10 times on data
trainer = flash.Trainer(max_epochs=10)

# 5. Train the model
trainer.fit(model, datamodule=datamodule)

# 6. Test model
trainer.test()

# 7. Save it!
trainer.save_checkpoint("tabular_classification_model.pt")

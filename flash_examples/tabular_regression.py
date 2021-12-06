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
from flash.tabular import TabularRegressionData, TabularRegressor

# 1. Create the DataModule
download_data("https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv", "./data")

datamodule = TabularRegressionData.from_csv(
    categorical_fields=["Seasons", "Holiday", "Functioning Day"],
    numerical_fields=[
        "Hour",
        "Temperature(C)",
        "Humidity(%)",
        "Wind speed (m/s)",
        "Visibility (10m)",
        "Dew point temperature(C)",
        "Solar Radiation (MJ/m2)",
        "Rainfall(mm)",
        "Snowfall (cm)",
    ],
    target_fields="Rented Bike Count",
    train_file="data/SeoulBikeData.csv",
    val_split=0.1,
)

# 2. Build the task
model = TabularRegressor.from_data(datamodule, learning_rate=0.1)

# 3. Create the trainer and train the model
trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())
trainer.fit(model, datamodule=datamodule)

# 4. Generate predictions from a CSV
datamodule = TabularRegressionData.from_csv(predict_file="data/SeoulBikeData.csv")
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("tabular_regression_model.pt")

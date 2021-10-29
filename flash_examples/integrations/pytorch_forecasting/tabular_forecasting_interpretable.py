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
from flash.core.integrations.pytorch_forecasting import convert_predictions
from flash.core.utilities.imports import example_requires
from flash.tabular.forecasting import TabularForecaster, TabularForecastingData

example_requires(["tabular", "matplotlib"])

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pytorch_forecasting.data import NaNLabelEncoder  # noqa: E402
from pytorch_forecasting.data.examples import generate_ar_data  # noqa: E402

# Example based on this tutorial: https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/ar.html
# 1. Create the DataModule
data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100, seed=42)
data["static"] = 2
data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")

max_prediction_length = 20

training_cutoff = data["time_idx"].max() - max_prediction_length

datamodule = TabularForecastingData.from_data_frame(
    time_idx="time_idx",
    target="value",
    categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    group_ids=["series"],
    # only unknown variable is "value" - and N-Beats can also not take any additional variables
    time_varying_unknown_reals=["value"],
    max_encoder_length=60,
    max_prediction_length=max_prediction_length,
    train_data_frame=data[lambda x: x.time_idx <= training_cutoff],
    val_data_frame=data,
    batch_size=32,
)

# 2. Build the task
model = TabularForecaster(datamodule.parameters, backbone="n_beats", widths=[32, 512], backcast_loss_ratio=0.1)

# 3. Create the trainer and train the model
trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count(), gradient_clip_val=0.01)
trainer.fit(model, datamodule=datamodule)

# 4. Generate predictions
predictions = model.predict(data)
print(predictions)

# Convert predictions
predictions, inputs = convert_predictions(predictions)

# Plot predictions
for idx in range(10):  # plot 10 examples
    model.pytorch_forecasting_model.plot_prediction(inputs, predictions, idx=idx, add_loss_to_title=True)

# Plot interpretation
for idx in range(10):  # plot 10 examples
    model.pytorch_forecasting_model.plot_interpretation(inputs, predictions, idx=idx)

# Show the plots!
plt.show()

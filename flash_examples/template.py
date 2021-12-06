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
import numpy as np
import torch
from sklearn import datasets

import flash
from flash.template import TemplateData, TemplateSKLearnClassifier

# 1. Create the DataModule
datamodule = TemplateData.from_sklearn(
    train_bunch=datasets.load_iris(),
    val_split=0.1,
)

# 2. Build the task
model = TemplateSKLearnClassifier(num_features=datamodule.num_features, num_classes=datamodule.num_classes)

# 3. Create the trainer and train the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.fit(model, datamodule=datamodule)

# 4. Classify a few examples
predictions = model.predict(
    [
        np.array([4.9, 3.0, 1.4, 0.2]),
        np.array([6.9, 3.2, 5.7, 2.3]),
        np.array([7.2, 3.0, 5.8, 1.6]),
    ],
    input="numpy",
)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("template_model.pt")

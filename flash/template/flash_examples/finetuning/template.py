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
from sklearn import datasets

import flash
from flash.core.classification import Labels
from flash.template import TemplateData, TemplateSKLearnClassifier

# 1. Download the data
data_bunch = datasets.load_iris()

# 2. Load the data
datamodule = TemplateData.from_sklearn(
    train_bunch=data_bunch,
    val_split=0.8,
)

# 3. Build the model
model = TemplateSKLearnClassifier(
    num_features=datamodule.num_features,
    num_classes=datamodule.num_classes,
    serializer=Labels(),
)

# 4. Create the trainer.
trainer = flash.Trainer(max_epochs=20)

# 5. Train the model
trainer.fit(model, datamodule=datamodule)

# 6. Save it!
trainer.save_checkpoint("template_model.pt")

# 7. Classify a few examples
predictions = model.predict([
    np.array([4.9, 3.0, 1.4, 0.2]),
    np.array([6.9, 3.2, 5.7, 2.3]),
    np.array([7.2, 3.0, 5.8, 1.6]),
])
print(predictions)

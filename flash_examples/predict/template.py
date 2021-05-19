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

from flash import Trainer
from flash.template import TemplateData, TemplateSKLearnClassifier

# 1. Download the data
data_bunch = datasets.load_iris()

# 2. Load the model from a checkpoint
model = TemplateSKLearnClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/template_model.pt")

# 3. Classify a few examples
predictions = model.predict([
    np.array([4.9, 3.0, 1.4, 0.2]),
    np.array([6.9, 3.2, 5.7, 2.3]),
    np.array([7.2, 3.0, 5.8, 1.6]),
])
print(predictions)

# 4. Or generate predictions from a whole dataset!
datamodule = TemplateData.from_sklearn(predict_bunch=data_bunch)

predictions = Trainer().predict(model, datamodule=datamodule)
print(predictions)

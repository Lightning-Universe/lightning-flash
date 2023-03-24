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
import hummingbird.ml
import sklearn.datasets

from flash.core.serve import Composition, expose, ModelComponent
from flash.core.serve.types import Number, Table

feature_names = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]


class PricePrediction(ModelComponent):
    def __init__(self, model):  # skipcq: PYL-W0621
        self.model = model

    @expose(inputs={"table": Table(column_names=feature_names)}, outputs={"pred": Number()})
    def predict(self, table):
        return self.model(table)


data = sklearn.datasets.load_boston()
model = sklearn.linear_model.LinearRegression()
model.fit(data.data, data.target)

model = hummingbird.ml.convert(model, "torch", test_input=data.data[0:1]).model
comp = PricePrediction(model)
composit = Composition(comp=comp)
composit.serve()

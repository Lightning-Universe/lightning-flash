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
import pandas as pd
import requests
from sklearn.datasets import load_boston

boston = load_boston()
data = pd.DataFrame(boston.data[0:1])
data.columns = boston.feature_names
body = {"session": "UUID", "payload": {"table": {"features": data.to_dict()}}}
resp = requests.post("http://127.0.0.1:8000/predict", json=body)
print(resp.json())

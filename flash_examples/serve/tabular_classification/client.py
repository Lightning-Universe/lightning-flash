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

from flash.core.data.utils import download_data

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", "data/")

df = pd.read_csv("./data/titanic/predict.csv")
text = str(df.to_csv())
body = {"session": "UUID", "payload": {"inputs": {"data": text}}}
resp = requests.post("http://127.0.0.1:8000/predict", json=body)
print(resp.json())

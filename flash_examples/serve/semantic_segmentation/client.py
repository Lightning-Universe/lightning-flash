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
import base64
from pathlib import Path

import requests

import flash

with (Path(flash.ASSETS_ROOT) / "road.png").open("rb") as f:
    imgstr = base64.b64encode(f.read()).decode("UTF-8")

body = {"session": "UUID", "payload": {"inputs": {"data": imgstr}}}
resp = requests.post("http://127.0.0.1:8000/predict", json=body)
print(resp.json())

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
import os
import shutil
from glob import glob

import numpy as np

from flash.core.data import download_data


def hymenoptera_data_download(path: str, predict_size: int = 10):
    download_data("https://download.pytorch.org/tutorial/hymenoptera_data.zip", path)
    predict_folder = os.path.join(path, "hymenoptera_data/predict")
    if not os.path.exists(predict_folder):
        os.makedirs(predict_folder)
    if len(os.listdir(predict_folder)) > 0:
        return
    validation_image_paths = glob(os.path.join(path, "hymenoptera_data/val/*/*"))
    assert predict_size < len(validation_image_paths)
    indices = np.random.choice(range(len(validation_image_paths)), predict_size, replace=False)
    for index in indices:
        src = validation_image_paths[index]
        dst = os.path.join(predict_folder, src.split('/')[-1])
        shutil.copy(src, dst)

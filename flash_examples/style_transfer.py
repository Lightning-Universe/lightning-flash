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

import torch

import flash
from flash.core.data.utils import download_data
from flash.image.style_transfer import StyleTransfer, StyleTransferData

# 1. Create the DataModule
download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "./data")

datamodule = StyleTransferData.from_folders(train_folder="data/coco128/images/train2017")

# 2. Build the task
model = StyleTransfer(os.path.join(flash.ASSETS_ROOT, "starry_night.jpg"))

# 3. Create the trainer and train the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.fit(model, datamodule=datamodule)

# 4. Apply style transfer to a few images!
predictions = model.predict(
    [
        "data/coco128/images/train2017/000000000625.jpg",
        "data/coco128/images/train2017/000000000626.jpg",
        "data/coco128/images/train2017/000000000629.jpg",
    ]
)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("style_transfer_model.pt")

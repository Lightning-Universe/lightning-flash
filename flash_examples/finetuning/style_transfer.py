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
import sys

import flash
from flash.core.data.utils import download_data
from flash.core.utilities.imports import _PYSTICHE_AVAILABLE

if _PYSTICHE_AVAILABLE:
    import pystiche.demo

    from flash.image.style_transfer import StyleTransfer, StyleTransferData
else:
    print("Please, run `pip install pystiche`")
    sys.exit(1)

# 1. Download the data
download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")

# 2. Load the data
datamodule = StyleTransferData.from_folders(train_folder="data/coco128/images", batch_size=4)

# 3. Load the style image
style_image = pystiche.demo.images()["paint"].read(size=256)

# 4. Build the model
model = StyleTransfer(style_image)

# 5. Create the trainer
trainer = flash.Trainer(max_epochs=2)

# 6. Train the model
trainer.fit(model, datamodule=datamodule)

# 7. Save it!
trainer.save_checkpoint("style_transfer_model.pt")

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
import flash
from flash.core.utilities.imports import example_requires
from flash.image import KeypointDetectionData, KeypointDetector

example_requires("image")

import icedata  # noqa: E402

# 1. Create the DataModule
data_dir = icedata.biwi.load_data()

datamodule = KeypointDetectionData.from_icedata(
    train_folder=data_dir,
    val_split=0.1,
    parser=icedata.biwi.parser,
    batch_size=4,
)

# 2. Build the task
model = KeypointDetector(
    head="keypoint_rcnn",
    backbone="resnet18_fpn",
    num_keypoints=1,
    num_classes=datamodule.num_classes,
)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=1)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Detect objects in a few images!
datamodule = KeypointDetectionData.from_files(
    predict_files=[
        str(data_dir / "biwi_sample/images/0.jpg"),
        str(data_dir / "biwi_sample/images/1.jpg"),
        str(data_dir / "biwi_sample/images/10.jpg"),
    ],
    batch_size=4,
)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("keypoint_detection_model.pt")

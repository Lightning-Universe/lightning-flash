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
from functools import partial

import flash
from flash.core.utilities.imports import example_requires
from flash.image import InstanceSegmentation, InstanceSegmentationData

example_requires("image")

import icedata  # noqa: E402

# 1. Create the DataModule
data_dir = icedata.pets.load_data()

datamodule = InstanceSegmentationData.from_folders(
    train_folder=data_dir,
    val_split=0.1,
    parser=partial(icedata.pets.parser, mask=True),
)

# 2. Build the task
model = InstanceSegmentation(
    head="mask_rcnn",
    backbone="resnet18_fpn",
    num_classes=datamodule.num_classes,
)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=1)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Detect objects in a few images!
predict_datamodule = InstanceSegmentationData.from_files(
    predict_files=[
        str(data_dir / "images/yorkshire_terrier_9.jpg"),
        str(data_dir / "images/yorkshire_terrier_12.jpg"),
        str(data_dir / "images/yorkshire_terrier_13.jpg"),
    ]
)
predictions = trainer.predict(model, datamodule=predict_datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("instance_segmentation_model.pt")

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
from itertools import chain

import flash
from flash.core.integrations.fiftyone import visualize
from flash.core.utilities.imports import example_requires
from flash.image import ObjectDetectionData, ObjectDetector
from flash.image.detection.serialization import FiftyOneDetectionLabels

example_requires("image")

import icedata  # noqa: E402

# 1. Create the DataModule
data_dir = icedata.fridge.load_data()

datamodule = ObjectDetectionData.from_folders(
    train_folder=data_dir,
    predict_folder=data_dir,
    val_split=0.1,
    image_size=128,
    parser=icedata.fridge.parser,
)

# 2. Build the task
model = ObjectDetector(head="efficientdet", backbone="d0", num_classes=datamodule.num_classes, image_size=128)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=1)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Set the serializer and get some predictions
model.serializer = FiftyOneDetectionLabels(return_filepath=True)  # output FiftyOne format
predictions = trainer.predict(model, datamodule=datamodule)
predictions = list(chain.from_iterable(predictions))  # flatten batches

# 5. Visualize predictions in FiftyOne app
# Optional: pass `wait=True` to block execution until App is closed
session = visualize(predictions, wait=True)

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
import torch

import flash
from flash.core.utilities.imports import example_requires
from flash.image import FaceDetectionData, FaceDetector

example_requires("fastface")
import fastface as ff  # noqa: E402

# # 1. Create the DataModule
train_dataset = ff.dataset.FDDBDataset(source_dir="data/", phase="train")
val_dataset = ff.dataset.FDDBDataset(source_dir="data/", phase="val")

datamodule = FaceDetectionData.from_datasets(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=2)

# # 2. Build the task
model = FaceDetector(model="lffd_slim")

# # 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count(), fast_dev_run=True)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Detect faces in a few images!
predict_datamodule = FaceDetectionData.from_files(
    predict_files=[
        "data/2002/07/19/big/img_18.jpg",
        "data/2002/07/19/big/img_65.jpg",
        "data/2002/07/19/big/img_255.jpg",
    ]
)
predictions = trainer.predict(model, datamodule=predict_datamodule)
print(predictions)

# # 5. Save the model!
trainer.save_checkpoint("face_detection_model.pt")

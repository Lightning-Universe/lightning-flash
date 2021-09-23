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
from flash.core.data.data_module import DataModule
from flash.core.utilities.imports import _FASTFACE_AVAILABLE
from flash.image import FaceDetector
from flash.image.face_detection.data import FaceDetectionPreprocess

if _FASTFACE_AVAILABLE:
    import fastface as ff
else:
    raise ModuleNotFoundError("Please, pip install -e '.[image]'")

# 1. Create the DataModule
train_dataset = ff.dataset.FDDBDataset(source_dir="data/", phase="train")
val_dataset = ff.dataset.FDDBDataset(source_dir="data/", phase="val")

datamodule = DataModule.from_data_source(
    "fastface", train_data=train_dataset, val_data=val_dataset, preprocess=FaceDetectionPreprocess()
)

# 2. Build the task
model = FaceDetector(model="lffd_slim")

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, limit_train_batches=0.1, limit_val_batches=0.1)

trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Detect faces in a few images!
predictions = model.predict([
    "data/2002/07/19/big/img_18.jpg",
    "data/2002/07/19/big/img_65.jpg",
    "data/2002/07/19/big/img_255.jpg",
])
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("face_detection_model.pt")

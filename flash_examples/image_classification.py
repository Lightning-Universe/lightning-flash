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
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
    batch_size=4,
    transform_kwargs={"image_size": (196, 196), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
)

# 2. Build the task
model = ImageClassifier(backbone="resnet18", labels=datamodule.labels)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Predict what's on a few images! ants or bees?
datamodule = ImageClassificationData.from_files(
    predict_files=[
        "https://pl-flash-data.s3.amazonaws.com/images/ant_1.jpg",
        "https://pl-flash-data.s3.amazonaws.com/images/ant_2.jpg",
        "https://pl-flash-data.s3.amazonaws.com/images/bee_1.jpg",
    ],
    batch_size=3,
)
predictions = trainer.predict(model, datamodule=datamodule, output="labels")
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("image_classification_model.pt")

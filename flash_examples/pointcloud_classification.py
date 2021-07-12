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
from flash.core.pipeline import Pipeline
from flash.pointcloud import PointCloudClassificationData, PointCloudClassifier
from flash.pointcloud.classification.datasets import ShapenetDataset

# 1. Create the DataModule
# construct a dataset by specifying dataset_path
dataset = ShapenetDataset("data")

datamodule = PointCloudClassificationData.from_datasets(
    train_dataset=dataset.get_split("training"),
    val_dataset=dataset.get_split("val"),
)

# 2. Build the task
model = PointCloudClassifier(backbone="randlanet_s3dis", num_classes=datamodule.num_classes)
pipeline = Pipeline(model, datamodule)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3)
trainer.finetune(pipeline, strategy="freeze")

# 4. Predict what's on a few PointClouds! ants or bees?
predictions = model.predict([
    "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
    "data/hymenoptera_data/val/bees/590318879_68cf112861.jpg",
    "data/hymenoptera_data/val/ants/540543309_ddbb193ee5.jpg",
])
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("PointCloud_classification_model.pt")

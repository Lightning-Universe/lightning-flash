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
from flash.core.data.utils import download_data
from flash.core.pipeline import Pipeline
from flash.pointcloud import PointCloudSegmentation, PointCloudSegmentationData

# 1. Create the DataModule
# Dataset Credit: http://www.semantic-kitti.org/
download_data("https://pl-flash-data.s3.amazonaws.com/SemanticKittiSmall.zip", "data/")

datamodule = PointCloudSegmentationData.from_folders(
    train_folder="data/SemanticKittiSmall/train",
    val_folder='data/SemanticKittiSmall/val',
    num_workers=2,
    batch_size=2,
)

# 2. Build the task
model = PointCloudSegmentation(backbone="randlanet_semantic_kitti", num_classes=datamodule.num_classes)
pipeline = Pipeline(model, datamodule)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(fast_dev_run=True, gpus=1)
trainer.fit(pipeline)

# 4. Predict what's within a few PointClouds?
predictions = pipeline.predict([
    "data/SemanticKittiSmall/val/02/scans/000000.bin",
    "data/SemanticKittiSmall/val/02/scans/000001.bin",
])

# 5. Save the model!
trainer.save_checkpoint("pointcloud_segmentation_model.pt")

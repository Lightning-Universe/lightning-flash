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

import fiftyone as fo
import torch

import flash
from flash.core.classification import FiftyOneLabels, Labels
from flash.core.data.utils import download_data
from flash.core.finetuning import FreezeUnfreeze
from flash.image import ImageClassificationData, ImageClassifier

# 1 Download data
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip")

# 2 Load data into FiftyOne
train_dataset = fo.Dataset.from_dir(
    dataset_dir="data/hymenoptera_data/train/",
    dataset_type=fo.types.ImageClassificationDirectoryTree,
)
val_dataset = fo.Dataset.from_dir(
    dataset_dir="data/hymenoptera_data/val/",
    dataset_type=fo.types.ImageClassificationDirectoryTree,
)
test_dataset = fo.Dataset.from_dir(
    dataset_dir="data/hymenoptera_data/test/",
    dataset_type=fo.types.ImageClassificationDirectoryTree,
)

# 3 Load data into Flash
datamodule = ImageClassificationData.from_fiftyone(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
)

# 4 Fine tune model
model = ImageClassifier(
    backbone="resnet18",
    num_classes=datamodule.num_classes,
    serializer=Labels(),
)
trainer = flash.Trainer(
    max_epochs=1,
    gpus=torch.cuda.device_count(),
    limit_train_batches=1,
    limit_val_batches=1,
)
trainer.finetune(
    model,
    datamodule=datamodule,
    strategy=FreezeUnfreeze(unfreeze_epoch=1),
)
trainer.save_checkpoint("image_classification_model.pt")

# 5 Predict from checkpoint on data with ground truth
model = ImageClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/image_classification_model.pt")
model.serializer = FiftyOneLabels(return_filepath=False)  # output FiftyOne format
datamodule = ImageClassificationData.from_fiftyone(predict_dataset=test_dataset)
predictions = trainer.predict(model, datamodule=datamodule)
predictions = list(chain.from_iterable(predictions))  # flatten batches

# 6 Add predictions to dataset
test_dataset.set_values("predictions", predictions)

# 7 Evaluate your model
results = test_dataset.evaluate_classifications("predictions", gt_field="ground_truth", eval_key="eval")
results.print_report()
plot = results.plot_confusion_matrix()
plot.show()

# 8 Visualize results in the App
session = fo.launch_app(test_dataset)

# Optional: block execution until App is closed
session.wait()

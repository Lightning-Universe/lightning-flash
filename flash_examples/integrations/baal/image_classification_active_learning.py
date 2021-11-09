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
from flash.core.classification import Probabilities
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier
from flash.image.classification.integrations.baal import ActiveLearningDataModule, ActiveLearningLoop

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

# Implement the research use-case where we mask labels from labelled dataset.
datamodule = ActiveLearningDataModule(
    ImageClassificationData.from_folders(train_folder="data/hymenoptera_data/train/", batch_size=2),
    initial_num_labels=5,
    val_split=0.1,
)

# 2. Build the task
head = torch.nn.Sequential(
    torch.nn.Dropout(p=0.1),
    torch.nn.Linear(512, datamodule.num_classes),
)
model = ImageClassifier(backbone="resnet18", head=head, num_classes=datamodule.num_classes, output=Probabilities())


# 3.1 Create the trainer
trainer = flash.Trainer(max_epochs=3)

# 3.2 Create the active learning loop and connect it to the trainer
active_learning_loop = ActiveLearningLoop(label_epoch_frequency=1)
active_learning_loop.connect(trainer.fit_loop)
trainer.fit_loop = active_learning_loop

# 3.3 Finetune
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Predict what's on a few images! ants or bees?
predictions = model.predict("data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg")
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("image_classification_model.pt")

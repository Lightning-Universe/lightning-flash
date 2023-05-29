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
import torch
from flash.text import TextClassificationData, TextEmbedder

# 1. Create the DataModule
datamodule = TextClassificationData.from_lists(
    predict_data=[
        "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
        "The worst movie in the history of cinema.",
        "I come from Bulgaria where it 's almost impossible to have a tornado.",
    ],
    batch_size=4,
)

# 2. Load a previously trained TextEmbedder
model = TextEmbedder(backbone="sentence-transformers/all-MiniLM-L6-v2")

# 3. Generate embeddings for the first 3 graphs
trainer = flash.Trainer(gpus=torch.cuda.device_count())
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

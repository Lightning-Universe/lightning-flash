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
from flash.text import TextGeneration, TextGenerationData

download_data(
    "https://raw.githubusercontent.com/adigoryl/Styled-Lyrics-Generator-GPT2/master/datasets" "/genius_lyrics_v2.csv",
    "./data/",
)

datamodule = TextGenerationData.from_csv(
    "lyrics",
    train_file="data/genius_lyrics_v2.csv",
    backbone="gpt2",
)

# 2. Build the task
model = TextGeneration()


# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

predictions = model.predict("Turgid dialogue, feeble characterization")

print(predictions)

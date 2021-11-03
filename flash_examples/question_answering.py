import pytorch_lightning as pl

pl.seed_everything(42)
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
from flash import Trainer
from flash.core.data.utils import download_data
from flash.text import QuestionAnsweringData, QuestionAnsweringTask

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/squad_tiny.zip", "./data/")

datamodule = QuestionAnsweringData.from_squad_v2(
    train_file="./data/squad_tiny/train.json",
    val_file="./data/squad_tiny/val.json",
)

# 2. Build the task
model = QuestionAnsweringTask()

# 3. Create the trainer and finetune the model
trainer = Trainer(max_epochs=3, limit_train_batches=1, limit_val_batches=1)
trainer.finetune(model, datamodule=datamodule)

# 4. Answer some Questions!
predictions = model.predict(
    {
        "id": ["56ddde6b9a695914005b9629", "56ddde6b9a695914005b9628"],
        "context": [
            """
        The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th
        and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse
        ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under
        their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations
        of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their
        descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct
        cultural and ethnic identity of the Normans emerged initially in the first half of the 10th
        century, and it continued to evolve over the succeeding centuries.
        """,
            """
        The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th
        and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse
        ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under
        their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations
        of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their
        descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct
        cultural and ethnic identity of the Normans emerged initially in the first half of the 10th
        century, and it continued to evolve over the succeeding centuries.
        """,
        ],
        "question": ["When were the Normans in Normandy?", "In what country is Normandy located?"],
    }
)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("question_answering_on_sqaud_v2.pt")

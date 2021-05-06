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

from flash import Trainer
from flash.data.utils import download_data
from flash.text import SummarizationData, SummarizationTask

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/xsum.zip", "data/")

# 2. Load the data
datamodule = SummarizationData.from_csv(
    "input",
    "target",
    train_file="data/xsum/train.csv",
    val_file="data/xsum/valid.csv",
    test_file="data/xsum/test.csv",
)

# 3. Build the model
model = SummarizationTask()

# 4. Create the trainer. Run once on data
trainer = Trainer(gpus=int(torch.cuda.is_available()), fast_dev_run=True)

# 5. Fine-tune the model
trainer.finetune(model, datamodule=datamodule)

# 6. Save it!
trainer.save_checkpoint("summarization_model_xsum.pt")

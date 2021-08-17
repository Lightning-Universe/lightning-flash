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
import time
from typing import Any

import psutil
import torch
from pytorch_lightning import Callback
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.types import STEP_OUTPUT

import flash
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier


class CUDACallback(Callback):
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if batch_idx == 1:
            # only start at the second batch
            # Reset the memory use counter
            torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
            torch.cuda.synchronize(trainer.root_gpu)
            self.start_time = time.time()

    def on_batch_end(self, trainer, pl_module) -> None:
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        pl_module.log("Peak Memory (GiB)", max_memory / 1000, prog_bar=True, on_step=True, sync_dist=True)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time
        virt_mem = psutil.virtual_memory()
        virt_mem = round((virt_mem.used / (1024 ** 3)), 2)
        swap = psutil.swap_memory()
        swap = round((swap.used / (1024 ** 3)), 2)

        max_memory = trainer.training_type_plugin.reduce(max_memory)
        epoch_time = trainer.training_type_plugin.reduce(epoch_time)
        virt_mem = trainer.training_type_plugin.reduce(virt_mem)
        swap = trainer.training_type_plugin.reduce(swap)

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak CUDA memory {max_memory:.2f} MiB")
        rank_zero_info(f"Average Peak Virtual memory {virt_mem:.2f} GiB")
        rank_zero_info(f"Average Peak Swap memory {swap:.2f} Gib")


if __name__ == "__main__":
    # 1. Create the DataModule
    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")

    datamodule = TextClassificationData.from_csv(
        "review",
        "sentiment",
        train_file="data/imdb/train.csv",
        val_file="data/imdb/valid.csv",
        backbone="facebook/bart-large",
        batch_size=4,
    )

    # 2. Build the task
    model = TextClassifier(backbone="facebook/bart-large", num_classes=datamodule.num_classes, enable_ort=False)

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(
        max_epochs=1,
        plugins=DeepSpeedPlugin(stage=1),
        callbacks=CUDACallback(),
        precision=16,
        accelerator="ddp",
        gpus=4,
        limit_val_batches=0,
        limit_test_batches=0,
    )
    trainer.fit(model, datamodule=datamodule)

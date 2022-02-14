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
from flash.core.utilities.flash_cli import FlashCLI
from flash.image.face_detection.data import FaceDetectionData
from flash.image.face_detection.model import FaceDetector

__all__ = ["face_detection"]


def from_fddb(
    batch_size: int = 1,
    **data_module_kwargs,
) -> FaceDetectionData:
    """Downloads and loads the FDDB data set."""
    import fastface as ff

    train_dataset = ff.dataset.FDDBDataset(source_dir="data/", phase="train")
    val_dataset = ff.dataset.FDDBDataset(source_dir="data/", phase="val")

    return FaceDetectionData.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        **data_module_kwargs,
    )


def face_detection():
    """Detect faces in images."""
    cli = FlashCLI(
        FaceDetector,
        FaceDetectionData,
        default_datamodule_builder=from_fddb,
        default_arguments={
            "trainer.max_epochs": 3,
        },
    )

    cli.trainer.save_checkpoint("face_detection_model.pt")


if __name__ == "__main__":
    face_detection()

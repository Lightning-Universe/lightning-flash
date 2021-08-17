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
from functools import partial
from typing import Callable, Optional

from flash.core.utilities.flash_cli import FlashCLI
from flash.core.utilities.imports import _ICEDATA_AVAILABLE, requires_extras
from flash.image import InstanceSegmentation, InstanceSegmentationData

if _ICEDATA_AVAILABLE:
    import icedata

__all__ = ["instance_segmentation"]


@requires_extras("image")
def from_pets(
    val_split: float = 0.1,
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    parser: Optional[Callable] = None,
    **preprocess_kwargs,
) -> InstanceSegmentationData:
    """Downloads and loads the pets data set from icedata."""
    data_dir = icedata.pets.load_data()

    if parser is None:
        parser = partial(icedata.pets.parser, mask=True)

    return InstanceSegmentationData.from_folders(
        train_folder=data_dir,
        val_split=val_split,
        batch_size=batch_size,
        num_workers=num_workers,
        parser=parser,
        **preprocess_kwargs,
    )


def instance_segmentation():
    """Segment object instances in images."""
    cli = FlashCLI(
        InstanceSegmentation,
        InstanceSegmentationData,
        default_datamodule_builder=from_pets,
        default_arguments={
            "trainer.max_epochs": 3,
        },
    )

    cli.trainer.save_checkpoint("instance_segmentation_model.pt")


if __name__ == "__main__":
    instance_segmentation()

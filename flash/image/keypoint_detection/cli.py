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
from typing import Any, Callable, Dict, Optional

from flash.core.utilities.flash_cli import FlashCLI
from flash.core.utilities.imports import _ICEDATA_AVAILABLE, requires
from flash.image import KeypointDetectionData, KeypointDetector

if _ICEDATA_AVAILABLE:
    import icedata

__all__ = ["keypoint_detection"]


@requires("image")
def from_biwi(
    train_folder: Optional[str] = None,
    train_ann_file: Optional[str] = None,
    val_folder: Optional[str] = None,
    val_ann_file: Optional[str] = None,
    test_folder: Optional[str] = None,
    test_ann_file: Optional[str] = None,
    predict_folder: Optional[str] = None,
    val_split: float = 0.1,
    parser: Optional[Callable] = None,
    transform_kwargs: Optional[Dict[str, Any]] = None,
    batch_size: int = 1,
    **data_module_kwargs,
) -> KeypointDetectionData:
    """Downloads and loads the BIWI data set from icedata."""
    data_dir = icedata.biwi.load_data()

    if parser is None:
        parser = icedata.biwi.parser

    return KeypointDetectionData.from_icedata(
        train_folder=train_folder or data_dir,
        train_ann_file=train_ann_file,
        val_folder=val_folder,
        val_ann_file=val_ann_file,
        test_folder=test_folder,
        test_ann_file=test_ann_file,
        predict_folder=predict_folder,
        val_split=val_split,
        transform_kwargs=dict(image_size=(128, 128)) if transform_kwargs is None else transform_kwargs,
        batch_size=batch_size,
        parser=parser,
        **data_module_kwargs,
    )


def keypoint_detection():
    """Detect keypoints in images."""
    cli = FlashCLI(
        KeypointDetector,
        KeypointDetectionData,
        default_datamodule_builder=from_biwi,
        default_arguments={
            "model.num_keypoints": 1,
            "trainer.max_epochs": 3,
        },
    )

    cli.trainer.save_checkpoint("keypoint_detection_model.pt")


if __name__ == "__main__":
    keypoint_detection()

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

from flash.core.utilities.imports import _ICEVISION_AVAILABLE
from flash.image import InstanceSegmentation, InstanceSegmentationData

if _ICEVISION_AVAILABLE:
    import icedata
from unittest import mock

import pytest

from flash.__main__ import main
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _IMAGE_AVAILABLE


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _ICEVISION_AVAILABLE, reason="IceVision is not installed for testing")
def test_cli():
    cli_args = ["flash", "instance_segmentation", "--trainer.fast_dev_run", "True"]
    with mock.patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _ICEVISION_AVAILABLE, reason="IceVision is not installed for testing")
def test_instance_segmentation_inference():
    """Test to ensure that inference runs with instance segmentation from input paths."""

    data_dir = icedata.pets.load_data()

    datamodule = InstanceSegmentationData.from_folders(
        train_folder=data_dir,
        val_split=0.1,
        parser=partial(icedata.pets.parser, mask=True),
    )

    model = InstanceSegmentation(
        head="mask_rcnn",
        backbone="resnet18_fpn",
        num_classes=datamodule.num_classes,
    )
    from flash.image.instance_segmentation.data import InstanceSegmentationPreprocess

    model._preprocess = InstanceSegmentationPreprocess()

    predictions = model.predict(
        [
            str(data_dir / "images/yorkshire_terrier_9.jpg"),
            str(data_dir / "images/yorkshire_terrier_12.jpg"),
            str(data_dir / "images/yorkshire_terrier_13.jpg"),
        ]
    )
    assert len(predictions) == 3

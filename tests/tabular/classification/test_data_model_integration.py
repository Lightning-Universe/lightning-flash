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
import pytest
import pytorch_lightning as pl

from flash.core.utilities.imports import _TABULAR_AVAILABLE
from flash.tabular import TabularClassificationData, TabularClassifier
from tests.helpers.utils import _TABULAR_TESTING

if _TABULAR_AVAILABLE:
    import pandas as pd

    TEST_DF_1 = pd.DataFrame(
        data={
            "category": ["a", "b", "c", "a", None, "c"],
            "scalar_a": [0.0, 1.0, 2.0, 3.0, None, 5.0],
            "scalar_b": [5.0, 4.0, 3.0, 2.0, None, 1.0],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
@pytest.mark.parametrize("backbone", ["tabnet", "tabtransformer", "fttransformer", "autoint",
                                      "node", "category_embedding"])
def test_classification(backbone, tmpdir):

    train_data_frame = TEST_DF_1.copy()
    val_data_frame = TEST_DF_1.copy()
    test_data_frame = TEST_DF_1.copy()
    data = TabularClassificationData.from_data_frame(
        categorical_fields=["category"],
        numerical_fields=["scalar_a", "scalar_b"],
        target_fields="label",
        train_data_frame=train_data_frame,
        val_data_frame=val_data_frame,
        test_data_frame=test_data_frame,
        num_workers=0,
        batch_size=2,
    )
    model = TabularClassifier.from_data(datamodule=data, backbone=backbone)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, data)

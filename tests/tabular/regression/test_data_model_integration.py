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

from flash.core.utilities.imports import _TOPIC_TABULAR_AVAILABLE
from flash.tabular import TabularRegressionData, TabularRegressor

if _TOPIC_TABULAR_AVAILABLE:
    import pandas as pd

    TEST_DICT = {
        "category": ["a", "b", "c", "a", None, "c"],
        "scalar_a": [0.0, 1.0, 2.0, 3.0, None, 5.0],
        "scalar_b": [5.0, 4.0, 3.0, 2.0, None, 1.0],
        "label": [0.0, 1.0, 2.0, 1.0, 0.0, 1.0],
    }

    TEST_LIST = [
        {"category": "a", "scalar_a": 0.0, "scalar_b": 5.0, "label": 0},
        {"category": "b", "scalar_a": 1.0, "scalar_b": 4.0, "label": 1},
        {"category": "c", "scalar_a": 2.0, "scalar_b": 3.0, "label": 0},
        {"category": "a", "scalar_a": 3.0, "scalar_b": 2.0, "label": 1},
        {"category": None, "scalar_a": None, "scalar_b": None, "label": 0},
        {"category": "c", "scalar_a": 5.0, "scalar_b": 1.0, "label": 1},
    ]

    TEST_DF = pd.DataFrame(data=TEST_DICT)


@pytest.mark.skipif(not _TOPIC_TABULAR_AVAILABLE, reason="tabular libraries aren't installed.")
@pytest.mark.parametrize(
    ("backbone", "fields"),
    [
        ("tabnet", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("tabtransformer", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("fttransformer", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("autoint", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("node", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        # ("category_embedding",  # todo: seems to be bug in tabular
        #  {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        # No categorical / numerical fields
        ("tabnet", {"categorical_fields": ["category"]}),
        ("tabnet", {"numerical_fields": ["scalar_a", "scalar_b"]}),
    ],
)
def test_regression_data_frame(backbone, fields, tmpdir):
    train_data_frame = TEST_DF.copy()
    val_data_frame = TEST_DF.copy()
    test_data_frame = TEST_DF.copy()
    data = TabularRegressionData.from_data_frame(
        **fields,
        target_field="label",
        train_data_frame=train_data_frame,
        val_data_frame=val_data_frame,
        test_data_frame=test_data_frame,
        num_workers=0,
        batch_size=2,
    )
    model = TabularRegressor.from_data(datamodule=data, backbone=backbone)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, data)


@pytest.mark.skipif(not _TOPIC_TABULAR_AVAILABLE, reason="tabular libraries aren't installed.")
@pytest.mark.parametrize(
    ("backbone", "fields"),
    [
        ("tabnet", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("tabtransformer", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("fttransformer", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("autoint", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("node", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        # ("category_embedding",  # todo: seems to be bug in tabular
        #  {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        # No categorical / numerical fields
        ("tabnet", {"categorical_fields": ["category"]}),
        ("tabnet", {"numerical_fields": ["scalar_a", "scalar_b"]}),
    ],
)
def test_regression_dicts(backbone, fields, tmpdir):
    data = TabularRegressionData.from_dicts(
        **fields,
        target_field="label",
        train_dict=TEST_DICT,
        val_dict=TEST_DICT,
        test_dict=TEST_DICT,
        num_workers=0,
        batch_size=2,
    )
    model = TabularRegressor.from_data(datamodule=data, backbone=backbone)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, data)


@pytest.mark.skipif(not _TOPIC_TABULAR_AVAILABLE, reason="tabular libraries aren't installed.")
@pytest.mark.parametrize(
    ("backbone", "fields"),
    [
        ("tabnet", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("tabtransformer", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("fttransformer", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("autoint", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        ("node", {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        # ("category_embedding",  # todo: seems to be bug in tabular
        #  {"categorical_fields": ["category"], "numerical_fields": ["scalar_a", "scalar_b"]}),
        # No categorical / numerical fields
        ("tabnet", {"categorical_fields": ["category"]}),
        ("tabnet", {"numerical_fields": ["scalar_a", "scalar_b"]}),
    ],
)
def test_regression_lists(backbone, fields, tmpdir):
    data = TabularRegressionData.from_lists(
        **fields,
        target_field="label",
        train_list=TEST_LIST,
        val_list=TEST_LIST,
        test_list=TEST_LIST,
        num_workers=0,
        batch_size=2,
    )
    model = TabularRegressor.from_data(datamodule=data, backbone=backbone)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, data)

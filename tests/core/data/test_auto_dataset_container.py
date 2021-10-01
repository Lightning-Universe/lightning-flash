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
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.auto_dataset_container import FlashDatasetContainer
from flash.core.data.data_source import DataSource
from flash.core.registry import FlashRegistry


def test_auto_dataset_container():

    with pytest.raises(MisconfigurationException, match="should have ``data_sources_registry``"):
        _ = FlashDatasetContainer.from_datasets(train_dataset=range(10))

    registry = FlashRegistry("test")

    class TestLoader(FlashDatasetContainer):

        data_sources_registry = registry

    with pytest.raises(MisconfigurationException, match="should have ``data_sources_registry``"):
        _ = TestLoader.from_datasets(train_dataset=range(10))

    class TestEnum(LightningEnum):

        DATASETS = "dataset"
        BASE = "base"

    class TestLoader2(FlashDatasetContainer):

        data_sources_registry = registry
        default_data_source = TestEnum.DATASETS

    with pytest.raises(MisconfigurationException, match="should have ``data_sources_registry``"):
        _ = TestLoader2.from_datasets(train_dataset=range(10))

    class CustomDataSource(DataSource):
        def __init__(self, something: str):
            super().__init__()

    TestLoader2.register_data_source(data_source_cls=CustomDataSource, enum=TestEnum.BASE)

    with pytest.raises(MisconfigurationException, match="from_base"):
        _ = TestLoader2.from_datasets(train_dataset=range(10))

    container = TestLoader2.from_data_source(
        TestEnum.BASE, train_data=range(10), predict_data=range(10), something="something"
    )
    assert isinstance(container._data_source, CustomDataSource)
    assert container.train_dataset
    assert not container.val_dataset
    assert not container.test_dataset
    assert container.predict_dataset
    assert container._data_source_kwargs == dict(something="something")

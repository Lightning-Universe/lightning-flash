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

from flash.data.data_module import DataModule


def test_flash_special_arguments(tmpdir):

    class CustomDataModule(DataModule):

        test = 1

    dm = CustomDataModule()
    CustomDataModule.test = 2
    assert dm.test == 2

    class CustomDataModule2(DataModule):

        test = 1
        __flash_special_attr__ = ["test"]

    dm = CustomDataModule2()
    CustomDataModule2.test = 2
    assert dm.test == 1

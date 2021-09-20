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

from unittest import mock

from flash import DataModule


@mock.patch("flash.core.data.data_module.DataLoader")
def test_dataloaders_with_sampler(mock_dataloader):
    train_ds = val_ds = test_ds = "dataset"
    mock_sampler = mock.MagicMock()
    dm = DataModule(train_ds, val_ds, test_ds, num_workers=0, sampler=mock_sampler)
    assert dm.sampler is mock_sampler
    dl = dm.train_dataloader()
    kwargs = mock_dataloader.call_args[1]
    assert "sampler" in kwargs
    assert kwargs["sampler"] is mock_sampler.return_value
    for dl in [dm.val_dataloader(), dm.test_dataloader()]:
        kwargs = mock_dataloader.call_args[1]
        assert "sampler" not in kwargs

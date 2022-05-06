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
from typing import Any

import pytest
import torch

import flash
from flash import DataKeys
from flash.core.utilities.imports import _TABULAR_AVAILABLE, _TABULAR_TESTING
from flash.tabular.forecasting import TabularForecaster
from tests.helpers.task_tester import StaticDataset, TaskTester

if _TABULAR_AVAILABLE:
    from pytorch_forecasting.data import EncoderNormalizer, NaNLabelEncoder
else:
    EncoderNormalizer = object
    NaNLabelEncoder = object


class TestTabularForecaster(TaskTester):

    task = TabularForecaster
    # TODO: Reduce number of required parameters
    task_kwargs = {
        "parameters": {
            "time_idx": "time_idx",
            "target": "value",
            "group_ids": ["series"],
            "weight": None,
            "max_encoder_length": 60,
            "min_encoder_length": 60,
            "min_prediction_idx": 0,
            "min_prediction_length": 20,
            "max_prediction_length": 20,
            "static_categoricals": [],
            "static_reals": [],
            "time_varying_known_categoricals": [],
            "time_varying_known_reals": [],
            "time_varying_unknown_categoricals": [],
            "time_varying_unknown_reals": ["value"],
            "variable_groups": {},
            "constant_fill_strategy": {},
            "allow_missing_timesteps": False,
            "lags": {},
            "add_relative_time_idx": False,
            "add_target_scales": False,
            "add_encoder_length": False,
            "target_normalizer": EncoderNormalizer(),
            "categorical_encoders": {"series": NaNLabelEncoder(), "__group_id__series": NaNLabelEncoder()},
            "scalers": {},
            "randomize_length": None,
            "predict_mode": False,
            "data_sample": {
                "series": {0: 0},
                "time_idx": {0: 0},
                "value": {0: 0.0},
            },
        },
        "backbone": "n_beats",
        "backbone_kwargs": {"widths": [32, 512], "backcast_loss_ratio": 0.1},
    }
    cli_command = "tabular_forecasting"
    is_testing = _TABULAR_TESTING
    is_available = _TABULAR_AVAILABLE

    # # TODO: Resolve JIT issues
    scriptable = False
    traceable = False

    @property
    def example_forward_input(self):
        return {
            "encoder_cat": torch.empty(2, 60, 0, dtype=torch.int64),
            "encoder_cont": torch.zeros(2, 60, 1),
            "encoder_target": torch.zeros(2, 60),
            "encoder_lengths": torch.tensor([60, 60]),
            "decoder_cat": torch.empty(2, 20, 0, dtype=torch.int64),
            "decoder_cont": torch.zeros(2, 20, 1),
            "decoder_target": torch.zeros(2, 20),
            "decoder_lengths": torch.tensor([20, 20]),
            "decoder_time_idx": torch.ones(2, 20).long(),
            "groups": torch.tensor([[0], [1]]),
            "target_scale": torch.zeros(2, 2),
        }

    def check_forward_output(self, output: Any):
        assert isinstance(output["prediction"], torch.Tensor)
        assert output["prediction"].shape == torch.Size([2, 20])

    @property
    def example_train_sample(self):
        return {
            DataKeys.INPUT: {
                "x_cat": torch.empty(60, 0, dtype=torch.int64),
                "x_cont": torch.zeros(60, 1),
                "encoder_target": torch.zeros(60),
                "encoder_length": 60,
                "decoder_length": 20,
                "encoder_time_idx_start": 1,
                "groups": torch.zeros(1),
                "target_scale": torch.zeros(2),
            },
            DataKeys.TARGET: (torch.rand(20), None),
        }

    def test_testing_raises(self, tmpdir):
        """Tests that ``NotImplementedError`` is raised when attempting to perform a test pass."""
        dataset = StaticDataset(self.example_train_sample, 4)

        args = self.task_args
        kwargs = dict(**self.task_kwargs)
        model = self.task(*args, **kwargs)

        trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True)

        with pytest.raises(
            NotImplementedError, match="Backbones provided by PyTorch Forecasting don't support testing."
        ):
            trainer.test(model, model.process_test_dataset(dataset, batch_size=4))

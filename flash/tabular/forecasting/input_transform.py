# Copyright 2020 The PyTorch Lightning team and The HuggingFace Team. All rights reserved.

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

# `TabularForecastingInputCollateTransform._collate_fn` Adapted from
# https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/data/timeseries.py

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch.nn.utils import rnn

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.imports import _FORECASTING_AVAILABLE

if _FORECASTING_AVAILABLE:
    from pytorch_forecasting import TimeSeriesDataSet
else:
    TimeSeriesDataSet = object


@dataclass
class TabularForecastingInputCollateTransform(InputTransform):
    @staticmethod
    def _collate_fn(
        batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Collate function to combine items into mini-batch for dataloader.

        Args:
            batches (List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]): List of samples generated with
                :py:meth:`~__getitem__`.

        Returns:
            Tuple[Dict[str, torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]: minibatch
        """
        # collate function for dataloader
        # lengths
        encoder_lengths = torch.tensor([batch[0]["encoder_length"] for batch in batches], dtype=torch.long)
        decoder_lengths = torch.tensor([batch[0]["decoder_length"] for batch in batches], dtype=torch.long)

        # ids
        decoder_time_idx_start = (
            torch.tensor([batch[0]["encoder_time_idx_start"] for batch in batches], dtype=torch.long) + encoder_lengths
        )
        decoder_time_idx = decoder_time_idx_start.unsqueeze(1) + torch.arange(decoder_lengths.max()).unsqueeze(0)
        groups = torch.stack([batch[0]["groups"] for batch in batches])

        # features
        encoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        encoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        decoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        decoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        # target scale
        if isinstance(batches[0][0]["target_scale"], torch.Tensor):  # stack tensor
            target_scale = torch.stack([batch[0]["target_scale"] for batch in batches])
        elif isinstance(batches[0][0]["target_scale"], (list, tuple)):
            target_scale = []
            for idx in range(len(batches[0][0]["target_scale"])):
                if isinstance(batches[0][0]["target_scale"][idx], torch.Tensor):  # stack tensor
                    scale = torch.stack([batch[0]["target_scale"][idx] for batch in batches])
                else:
                    scale = torch.tensor([batch[0]["target_scale"][idx] for batch in batches], dtype=torch.float)
                target_scale.append(scale)
        else:  # convert to tensor
            target_scale = torch.tensor([batch[0]["target_scale"] for batch in batches], dtype=torch.float)

        # target and weight
        if isinstance(batches[0][1][0], (tuple, list)):
            target = [
                rnn.pad_sequence([batch[1][0][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
            encoder_target = [
                rnn.pad_sequence([batch[0]["encoder_target"][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
        else:
            target = rnn.pad_sequence([batch[1][0] for batch in batches], batch_first=True)
            encoder_target = rnn.pad_sequence([batch[0]["encoder_target"] for batch in batches], batch_first=True)

        if batches[0][1][1] is not None:
            weight = rnn.pad_sequence([batch[1][1] for batch in batches], batch_first=True)
        else:
            weight = None

        return (
            dict(
                encoder_cat=encoder_cat,
                encoder_cont=encoder_cont,
                encoder_target=encoder_target,
                encoder_lengths=encoder_lengths,
                decoder_cat=decoder_cat,
                decoder_cont=decoder_cont,
                decoder_target=target,
                decoder_lengths=decoder_lengths,
                decoder_time_idx=decoder_time_idx,
                groups=groups,
                target_scale=target_scale,
            ),
            (target, weight),
        )

    def collate(self) -> Callable:
        """Defines the transform to be applied on a list of sample to create a batch for all stages."""

        def _collate_fn(samples: Sequence):
            samples = [(sample[DataKeys.INPUT], sample[DataKeys.TARGET]) for sample in samples]
            batch = TabularForecastingInputCollateTransform._collate_fn(batches=samples)
            return {DataKeys.INPUT: batch[0], DataKeys.TARGET: batch[1]}

        return _collate_fn

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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor

from flash.core.data.callback import ControlFlow
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.utils import (
    _contains_any_tensor,
    convert_to_modules,
    CurrentFuncContext,
    CurrentRunningStageContext,
)

if TYPE_CHECKING:
    from flash.core.data.process import Deserializer, Preprocess, Serializer


class _Sequential(torch.nn.Module):
    """This class is used to chain 3 functions together for the _Preprocessor ``per_sample_transform`` function.

    1. ``pre_tensor_transform``
    2. ``to_tensor_transform``
    3. ``post_tensor_transform``
    """

    def __init__(
        self,
        preprocess: "Preprocess",
        pre_tensor_transform: Optional[Callable],
        to_tensor_transform: Optional[Callable],
        post_tensor_transform: Callable,
        stage: RunningStage,
        assert_contains_tensor: bool = False,
    ):
        super().__init__()
        self.preprocess = preprocess
        self.callback = ControlFlow(self.preprocess.callbacks)
        self.pre_tensor_transform = convert_to_modules(pre_tensor_transform)
        self.to_tensor_transform = convert_to_modules(to_tensor_transform)
        self.post_tensor_transform = convert_to_modules(post_tensor_transform)
        self.stage = stage
        self.assert_contains_tensor = assert_contains_tensor

        self._current_stage_context = CurrentRunningStageContext(stage, preprocess, reset=False)
        self._pre_tensor_transform_context = CurrentFuncContext("pre_tensor_transform", preprocess)
        self._to_tensor_transform_context = CurrentFuncContext("to_tensor_transform", preprocess)
        self._post_tensor_transform_context = CurrentFuncContext("post_tensor_transform", preprocess)

    def forward(self, sample: Any) -> Any:
        self.callback.on_load_sample(sample, self.stage)

        with self._current_stage_context:
            if self.pre_tensor_transform is not None:
                with self._pre_tensor_transform_context:
                    sample = self.pre_tensor_transform(sample)
                    self.callback.on_pre_tensor_transform(sample, self.stage)

            if self.to_tensor_transform is not None:
                with self._to_tensor_transform_context:
                    sample = self.to_tensor_transform(sample)
                    self.callback.on_to_tensor_transform(sample, self.stage)

                if self.assert_contains_tensor:
                    if not _contains_any_tensor(sample):
                        raise MisconfigurationException(
                            "When ``to_tensor_transform`` is overriden, "
                            "``DataPipeline`` expects the outputs to be ``tensors``"
                        )

            with self._post_tensor_transform_context:
                sample = self.post_tensor_transform(sample)
                self.callback.on_post_tensor_transform(sample, self.stage)

            return sample

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"\t(pre_tensor_transform): {str(self.pre_tensor_transform)}\n"
            f"\t(to_tensor_transform): {str(self.to_tensor_transform)}\n"
            f"\t(post_tensor_transform): {str(self.post_tensor_transform)}\n"
            f"\t(assert_contains_tensor): {str(self.assert_contains_tensor)}\n"
            f"\t(stage): {str(self.stage)}"
        )


class _DeserializeProcessor(torch.nn.Module):
    def __init__(
        self,
        deserializer: "Deserializer",
        preprocess: "Preprocess",
        pre_tensor_transform: Callable,
        to_tensor_transform: Callable,
    ):
        super().__init__()
        self.preprocess = preprocess
        self.callback = ControlFlow(self.preprocess.callbacks)
        self.deserializer = convert_to_modules(deserializer)
        self.pre_tensor_transform = convert_to_modules(pre_tensor_transform)
        self.to_tensor_transform = convert_to_modules(to_tensor_transform)

        self._current_stage_context = CurrentRunningStageContext(RunningStage.PREDICTING, preprocess, reset=False)
        self._pre_tensor_transform_context = CurrentFuncContext("pre_tensor_transform", preprocess)
        self._to_tensor_transform_context = CurrentFuncContext("to_tensor_transform", preprocess)

    def forward(self, sample: str):

        sample = self.deserializer(sample)

        with self._current_stage_context:
            with self._pre_tensor_transform_context:
                sample = self.pre_tensor_transform(sample)
                self.callback.on_pre_tensor_transform(sample, RunningStage.PREDICTING)

            with self._to_tensor_transform_context:
                sample = self.to_tensor_transform(sample)
                self.callback.on_to_tensor_transform(sample, RunningStage.PREDICTING)

        return sample


class _SerializeProcessor(torch.nn.Module):
    def __init__(
        self,
        serializer: "Serializer",
    ):
        super().__init__()
        self.serializer = convert_to_modules(serializer)

    def forward(self, sample):
        return self.serializer(sample)


class _Preprocessor(torch.nn.Module):
    """
    This class is used to encapsultate the following functions of a Preprocess Object:
    Inside a worker:
        per_sample_transform: Function to transform an individual sample
            Inside a worker, it is actually make of 3 functions:
                * pre_tensor_transform
                * to_tensor_transform
                * post_tensor_transform
        collate: Function to merge sample into a batch
        per_batch_transform: Function to transform an individual batch
            * per_batch_transform

    Inside main process:
        per_sample_transform: Function to transform an individual sample
            * per_sample_transform_on_device
        collate: Function to merge sample into a batch
        per_batch_transform: Function to transform an individual batch
            * per_batch_transform_on_device
    """

    def __init__(
        self,
        preprocess: "Preprocess",
        collate_fn: Callable,
        per_sample_transform: Union[Callable, _Sequential],
        per_batch_transform: Callable,
        stage: RunningStage,
        apply_per_sample_transform: bool = True,
        on_device: bool = False,
    ):
        super().__init__()
        self.preprocess = preprocess
        self.callback = ControlFlow(self.preprocess.callbacks)
        self.collate_fn = convert_to_modules(collate_fn)
        self.per_sample_transform = convert_to_modules(per_sample_transform)
        self.per_batch_transform = convert_to_modules(per_batch_transform)
        self.apply_per_sample_transform = apply_per_sample_transform
        self.stage = stage
        self.on_device = on_device

        extension = f"{'_on_device' if self.on_device else ''}"
        self._current_stage_context = CurrentRunningStageContext(stage, preprocess)
        self._per_sample_transform_context = CurrentFuncContext(f"per_sample_transform{extension}", preprocess)
        self._collate_context = CurrentFuncContext("collate", preprocess)
        self._per_batch_transform_context = CurrentFuncContext(f"per_batch_transform{extension}", preprocess)

    @staticmethod
    def _extract_metadata(
        samples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        metadata = [s.pop(DefaultDataKeys.METADATA, None) if isinstance(s, Mapping) else None for s in samples]
        return samples, metadata if any(m is not None for m in metadata) else None

    def forward(self, samples: Sequence[Any]) -> Any:
        # we create a new dict to prevent from potential memory leaks
        # assuming that the dictionary samples are stored in between and
        # potentially modified before the transforms are applied.
        if isinstance(samples, dict):
            samples = dict(samples.items())

        with self._current_stage_context:

            if self.apply_per_sample_transform:
                with self._per_sample_transform_context:
                    _samples = []

                    if isinstance(samples, Mapping):
                        samples = [samples]

                    for sample in samples:
                        sample = self.per_sample_transform(sample)
                        if self.on_device:
                            self.callback.on_per_sample_transform_on_device(sample, self.stage)
                        _samples.append(sample)

                samples = type(_samples)(_samples)

                with self._collate_context:
                    samples, metadata = self._extract_metadata(samples)
                    try:
                        samples = self.collate_fn(samples, metadata)
                    except TypeError:
                        samples = self.collate_fn(samples)
                    if metadata and isinstance(samples, dict):
                        samples[DefaultDataKeys.METADATA] = metadata
                    self.callback.on_collate(samples, self.stage)

            with self._per_batch_transform_context:
                samples = self.per_batch_transform(samples)
                if self.on_device:
                    self.callback.on_per_batch_transform_on_device(samples, self.stage)
                else:
                    self.callback.on_per_batch_transform(samples, self.stage)
            return samples

    def __str__(self) -> str:
        # todo: define repr function which would take object and string attributes to be shown
        return (
            "_Preprocessor:\n"
            f"\t(per_sample_transform): {str(self.per_sample_transform)}\n"
            f"\t(collate_fn): {str(self.collate_fn)}\n"
            f"\t(per_batch_transform): {str(self.per_batch_transform)}\n"
            f"\t(apply_per_sample_transform): {str(self.apply_per_sample_transform)}\n"
            f"\t(on_device): {str(self.on_device)}\n"
            f"\t(stage): {str(self.stage)}"
        )


class _Postprocessor(torch.nn.Module):
    """This class is used to encapsultate the following functions of a Postprocess Object:

    Inside main process:
        per_batch_transform: Function to transform a batch
        per_sample_transform: Function to transform an individual sample
        uncollate_fn: Function to split a batch into samples
        per_sample_transform: Function to transform an individual sample
        save_fn: Function to save all data
        save_per_sample: Function to save an individual sample
        is_serving: Whether the Postprocessor is used in serving mode.
    """

    def __init__(
        self,
        uncollate_fn: Callable,
        per_batch_transform: Callable,
        per_sample_transform: Callable,
        serializer: Optional[Callable],
        save_fn: Optional[Callable] = None,
        save_per_sample: bool = False,
        is_serving: bool = False,
    ):
        super().__init__()
        self.uncollate_fn = convert_to_modules(uncollate_fn)
        self.per_batch_transform = convert_to_modules(per_batch_transform)
        self.per_sample_transform = convert_to_modules(per_sample_transform)
        self.serializer = convert_to_modules(serializer)
        self.save_fn = convert_to_modules(save_fn)
        self.save_per_sample = convert_to_modules(save_per_sample)
        self.is_serving = is_serving

    @staticmethod
    def _extract_metadata(batch: Any) -> Tuple[Any, Optional[Any]]:
        metadata = None
        if isinstance(batch, Mapping) and DefaultDataKeys.METADATA in batch:
            metadata = batch.pop(DefaultDataKeys.METADATA, None)
        return batch, metadata

    def forward(self, batch: Sequence[Any]):
        batch, metadata = self._extract_metadata(batch)
        uncollated = self.uncollate_fn(self.per_batch_transform(batch))
        if metadata:
            for sample, sample_metadata in zip(uncollated, metadata):
                sample[DefaultDataKeys.METADATA] = sample_metadata

        final_preds = [self.per_sample_transform(sample) for sample in uncollated]

        if self.serializer is not None:
            final_preds = [self.serializer(sample) for sample in final_preds]

        if isinstance(uncollated, Tensor) and isinstance(final_preds[0], Tensor):
            final_preds = torch.stack(final_preds)
        else:
            final_preds = type(final_preds)(final_preds)

        if self.save_fn:
            if self.save_per_sample:
                for pred in final_preds:
                    self.save_fn(pred)
            else:
                self.save_fn(final_preds)
        return final_preds

    def __str__(self) -> str:
        return (
            "_Postprocessor:\n"
            f"\t(per_batch_transform): {str(self.per_batch_transform)}\n"
            f"\t(uncollate_fn): {str(self.uncollate_fn)}\n"
            f"\t(per_sample_transform): {str(self.per_sample_transform)}\n"
            f"\t(serializer): {str(self.serializer)}"
        )


def default_uncollate(batch: Any):
    """
    This function is used to uncollate a batch into samples.
    Examples:
        >>> a, b = default_uncollate(torch.rand((2,1)))
    """

    batch_type = type(batch)

    if isinstance(batch, Tensor):
        if len(batch.shape) == 0:  # 0 shape tensors
            return batch
        return list(torch.unbind(batch, 0))

    if isinstance(batch, Mapping):
        return [batch_type(dict(zip(batch, default_uncollate(t)))) for t in zip(*batch.values())]

    if isinstance(batch, tuple) and hasattr(batch, "_fields"):  # namedtuple
        return [batch_type(*default_uncollate(sample)) for sample in zip(*batch)]

    if isinstance(batch, Sequence) and not isinstance(batch, str):
        return [default_uncollate(sample) for sample in batch]

    return batch

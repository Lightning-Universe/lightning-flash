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
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from flash import Trainer
from flash.core.data.auto_dataset import IterableAutoDataset
from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import _StageOrchestrator, DataPipeline, DataPipelineState
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import _InputTransformProcessor, DefaultInputTransform, InputTransform
from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import _OutputTransformProcessor, OutputTransform
from flash.core.data.process import Deserializer
from flash.core.data.properties import ProcessState
from flash.core.data.states import PerBatchTransformOnDevice, ToTensorTransform
from flash.core.model import Task
from flash.core.utilities.imports import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from flash.core.utilities.stages import RunningStage
from tests.helpers.utils import _IMAGE_TESTING

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as T

if _PIL_AVAILABLE:
    from PIL import Image


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return torch.rand(1), torch.rand(1)

    def __len__(self) -> int:
        return 5


class TestDataPipelineState:
    @staticmethod
    def test_str():
        state = DataPipelineState()
        state.set_state(ProcessState())

        assert str(state) == (
            "DataPipelineState(state={<class 'flash.core.data.properties.ProcessState'>: ProcessState()})"
        )

    @staticmethod
    def test_get_state():
        state = DataPipelineState()
        assert state.get_state(ProcessState) is None


def test_data_pipeline_str():
    data_pipeline = DataPipeline(
        data_source=cast(Input, "data_source"),
        input_transform=cast(InputTransform, "input_transform"),
        output_transform=cast(OutputTransform, "output_transform"),
        output=cast(Output, "output"),
        deserializer=cast(Deserializer, "deserializer"),
    )

    expected = "data_source=data_source, deserializer=deserializer, "
    expected += "input_transform=input_transform, output_transform=output_transform, output=output"
    assert str(data_pipeline) == (f"DataPipeline({expected})")


@pytest.mark.parametrize("use_input_transform", [False, True])
@pytest.mark.parametrize("use_output_transform", [False, True])
def test_data_pipeline_init_and_assignement(use_input_transform, use_output_transform, tmpdir):
    class CustomModel(Task):
        def __init__(self, output_transform: Optional[OutputTransform] = None):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())
            self._output_transform = output_transform

        def train_dataloader(self) -> Any:
            return DataLoader(DummyDataset())

    class SubInputTransform(DefaultInputTransform):
        pass

    class SubOutputTransform(OutputTransform):
        pass

    data_pipeline = DataPipeline(
        input_transform=SubInputTransform() if use_input_transform else None,
        output_transform=SubOutputTransform() if use_output_transform else None,
    )
    assert isinstance(
        data_pipeline._input_transform_pipeline, SubInputTransform if use_input_transform else DefaultInputTransform
    )
    assert isinstance(data_pipeline._output_transform, SubOutputTransform if use_output_transform else OutputTransform)

    model = CustomModel(output_transform=OutputTransform())
    model.data_pipeline = data_pipeline
    # TODO: the line below should make the same effect but it's not
    # data_pipeline._attach_to_model(model)

    if use_input_transform:
        assert isinstance(model._input_transform, SubInputTransform)
    else:
        assert model._input_transform is None or isinstance(model._input_transform, InputTransform)

    if use_output_transform:
        assert isinstance(model._output_transform, SubOutputTransform)
    else:
        assert model._output_transform is None or isinstance(model._output_transform, OutputTransform)


def test_data_pipeline_is_overriden_and_resolve_function_hierarchy(tmpdir):
    class CustomInputTransform(DefaultInputTransform):
        def val_pre_tensor_transform(self, *_, **__):
            pass

        def predict_to_tensor_transform(self, *_, **__):
            pass

        def train_post_tensor_transform(self, *_, **__):
            pass

        def test_collate(self, *_, **__):
            pass

        def val_per_sample_transform_on_device(self, *_, **__):
            pass

        def train_per_batch_transform_on_device(self, *_, **__):
            pass

        def test_per_batch_transform_on_device(self, *_, **__):
            pass

    input_transform = CustomInputTransform()
    data_pipeline = DataPipeline(input_transform=input_transform)

    train_func_names: Dict[str, str] = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._input_transform_pipeline, RunningStage.TRAINING, InputTransform
        )
        for k in data_pipeline.INPUT_TRANSFORM_FUNCS
    }
    val_func_names: Dict[str, str] = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._input_transform_pipeline, RunningStage.VALIDATING, InputTransform
        )
        for k in data_pipeline.INPUT_TRANSFORM_FUNCS
    }
    test_func_names: Dict[str, str] = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._input_transform_pipeline, RunningStage.TESTING, InputTransform
        )
        for k in data_pipeline.INPUT_TRANSFORM_FUNCS
    }
    predict_func_names: Dict[str, str] = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._input_transform_pipeline, RunningStage.PREDICTING, InputTransform
        )
        for k in data_pipeline.INPUT_TRANSFORM_FUNCS
    }

    # pre_tensor_transform
    assert train_func_names["pre_tensor_transform"] == "pre_tensor_transform"
    assert val_func_names["pre_tensor_transform"] == "val_pre_tensor_transform"
    assert test_func_names["pre_tensor_transform"] == "pre_tensor_transform"
    assert predict_func_names["pre_tensor_transform"] == "pre_tensor_transform"

    # to_tensor_transform
    assert train_func_names["to_tensor_transform"] == "to_tensor_transform"
    assert val_func_names["to_tensor_transform"] == "to_tensor_transform"
    assert test_func_names["to_tensor_transform"] == "to_tensor_transform"
    assert predict_func_names["to_tensor_transform"] == "predict_to_tensor_transform"

    # post_tensor_transform
    assert train_func_names["post_tensor_transform"] == "train_post_tensor_transform"
    assert val_func_names["post_tensor_transform"] == "post_tensor_transform"
    assert test_func_names["post_tensor_transform"] == "post_tensor_transform"
    assert predict_func_names["post_tensor_transform"] == "post_tensor_transform"

    # collate
    assert train_func_names["collate"] == "collate"
    assert val_func_names["collate"] == "collate"
    assert test_func_names["collate"] == "test_collate"
    assert predict_func_names["collate"] == "collate"

    # per_sample_transform_on_device
    assert train_func_names["per_sample_transform_on_device"] == "per_sample_transform_on_device"
    assert val_func_names["per_sample_transform_on_device"] == "val_per_sample_transform_on_device"
    assert test_func_names["per_sample_transform_on_device"] == "per_sample_transform_on_device"
    assert predict_func_names["per_sample_transform_on_device"] == "per_sample_transform_on_device"

    # per_batch_transform_on_device
    assert train_func_names["per_batch_transform_on_device"] == "train_per_batch_transform_on_device"
    assert val_func_names["per_batch_transform_on_device"] == "per_batch_transform_on_device"
    assert test_func_names["per_batch_transform_on_device"] == "test_per_batch_transform_on_device"
    assert predict_func_names["per_batch_transform_on_device"] == "per_batch_transform_on_device"

    train_worker_input_transform_processor = data_pipeline.worker_input_transform_processor(RunningStage.TRAINING)
    val_worker_input_transform_processor = data_pipeline.worker_input_transform_processor(RunningStage.VALIDATING)
    test_worker_input_transform_processor = data_pipeline.worker_input_transform_processor(RunningStage.TESTING)
    predict_worker_input_transform_processor = data_pipeline.worker_input_transform_processor(RunningStage.PREDICTING)

    _seq = train_worker_input_transform_processor.per_sample_transform
    assert _seq.pre_tensor_transform.func == input_transform.pre_tensor_transform
    assert _seq.to_tensor_transform.func == input_transform.to_tensor_transform
    assert _seq.post_tensor_transform.func == input_transform.train_post_tensor_transform
    assert train_worker_input_transform_processor.collate_fn.func == input_transform.collate
    assert train_worker_input_transform_processor.per_batch_transform.func == input_transform.per_batch_transform

    _seq = val_worker_input_transform_processor.per_sample_transform
    assert _seq.pre_tensor_transform.func == input_transform.val_pre_tensor_transform
    assert _seq.to_tensor_transform.func == input_transform.to_tensor_transform
    assert _seq.post_tensor_transform.func == input_transform.post_tensor_transform
    assert val_worker_input_transform_processor.collate_fn.func == DataPipeline._identity
    assert val_worker_input_transform_processor.per_batch_transform.func == input_transform.per_batch_transform

    _seq = test_worker_input_transform_processor.per_sample_transform
    assert _seq.pre_tensor_transform.func == input_transform.pre_tensor_transform
    assert _seq.to_tensor_transform.func == input_transform.to_tensor_transform
    assert _seq.post_tensor_transform.func == input_transform.post_tensor_transform
    assert test_worker_input_transform_processor.collate_fn.func == input_transform.test_collate
    assert test_worker_input_transform_processor.per_batch_transform.func == input_transform.per_batch_transform

    _seq = predict_worker_input_transform_processor.per_sample_transform
    assert _seq.pre_tensor_transform.func == input_transform.pre_tensor_transform
    assert _seq.to_tensor_transform.func == input_transform.predict_to_tensor_transform
    assert _seq.post_tensor_transform.func == input_transform.post_tensor_transform
    assert predict_worker_input_transform_processor.collate_fn.func == input_transform.collate
    assert predict_worker_input_transform_processor.per_batch_transform.func == input_transform.per_batch_transform


class CustomInputTransform(DefaultInputTransform):
    def train_per_sample_transform(self, *_, **__):
        pass

    def train_per_batch_transform_on_device(self, *_, **__):
        pass

    def test_per_sample_transform(self, *_, **__):
        pass

    def test_per_batch_transform(self, *_, **__):
        pass

    def test_per_sample_transform_on_device(self, *_, **__):
        pass

    def test_per_batch_transform_on_device(self, *_, **__):
        pass

    def val_per_batch_transform(self, *_, **__):
        pass

    def val_per_sample_transform_on_device(self, *_, **__):
        pass

    def predict_per_sample_transform(self, *_, **__):
        pass

    def predict_per_sample_transform_on_device(self, *_, **__):
        pass

    def predict_per_batch_transform_on_device(self, *_, **__):
        pass


def test_data_pipeline_predict_worker_input_transform_processor_and_device_input_transform_processor():

    input_transform = CustomInputTransform()
    data_pipeline = DataPipeline(input_transform=input_transform)

    data_pipeline.worker_input_transform_processor(RunningStage.TRAINING)
    with pytest.raises(MisconfigurationException, match="are mutually exclusive"):
        data_pipeline.worker_input_transform_processor(RunningStage.VALIDATING)
    with pytest.raises(MisconfigurationException, match="are mutually exclusive"):
        data_pipeline.worker_input_transform_processor(RunningStage.TESTING)
    data_pipeline.worker_input_transform_processor(RunningStage.PREDICTING)


def test_detach_input_transform_from_model(tmpdir):
    class CustomModel(Task):
        def __init__(self, output_transform: Optional[OutputTransform] = None):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())
            self._output_transform = output_transform

        def train_dataloader(self) -> Any:
            return DataLoader(DummyDataset())

    input_transform = CustomInputTransform()
    data_pipeline = DataPipeline(input_transform=input_transform)
    model = CustomModel()
    model.data_pipeline = data_pipeline

    assert model.train_dataloader().collate_fn == default_collate
    assert model.transfer_batch_to_device.__self__ == model
    model.on_train_dataloader()
    assert isinstance(model.train_dataloader().collate_fn, _InputTransformProcessor)
    assert isinstance(model.transfer_batch_to_device, _StageOrchestrator)
    model.on_fit_end()
    assert model.transfer_batch_to_device.__self__ == model
    assert model.train_dataloader().collate_fn == default_collate


class TestInputTransform(DefaultInputTransform):
    def train_per_sample_transform(self, *_, **__):
        pass

    def train_per_batch_transform_on_device(self, *_, **__):
        pass

    def test_per_sample_transform(self, *_, **__):
        pass

    def test_per_sample_transform_on_device(self, *_, **__):
        pass

    def test_per_batch_transform_on_device(self, *_, **__):
        pass

    def val_per_sample_transform_on_device(self, *_, **__):
        pass

    def predict_per_sample_transform(self, *_, **__):
        pass

    def predict_per_sample_transform_on_device(self, *_, **__):
        pass

    def predict_per_batch_transform_on_device(self, *_, **__):
        pass


def test_attaching_datapipeline_to_model(tmpdir):
    class SubInputTransform(DefaultInputTransform):
        pass

    input_transform = SubInputTransform()
    data_pipeline = DataPipeline(input_transform=input_transform)

    class CustomModel(Task):
        def __init__(self):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())
            self._output_transform = OutputTransform()

        def training_step(self, batch: Any, batch_idx: int) -> Any:
            pass

        def validation_step(self, batch: Any, batch_idx: int) -> Any:
            pass

        def test_step(self, batch: Any, batch_idx: int) -> Any:
            pass

        def train_dataloader(self) -> Any:
            return DataLoader(DummyDataset())

        def val_dataloader(self) -> Any:
            return DataLoader(DummyDataset())

        def test_dataloader(self) -> Any:
            return DataLoader(DummyDataset())

        def predict_dataloader(self) -> Any:
            return DataLoader(DummyDataset())

    class TestModel(CustomModel):

        stages = [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]
        on_train_start_called = False
        on_val_start_called = False
        on_test_start_called = False
        on_predict_start_called = False

        def on_fit_start(self):
            assert self.predict_step.__self__ == self
            self._saved_predict_step = self.predict_step

        @staticmethod
        def _compare_pre_processor(p1, p2):
            p1_seq = p1.per_sample_transform
            p2_seq = p2.per_sample_transform
            assert p1_seq.pre_tensor_transform.func == p2_seq.pre_tensor_transform.func
            assert p1_seq.to_tensor_transform.func == p2_seq.to_tensor_transform.func
            assert p1_seq.post_tensor_transform.func == p2_seq.post_tensor_transform.func
            assert p1.collate_fn.func == p2.collate_fn.func
            assert p1.per_batch_transform.func == p2.per_batch_transform.func

        @staticmethod
        def _assert_stage_orchestrator_state(
            stage_mapping: Dict, current_running_stage: RunningStage, cls=_InputTransformProcessor
        ):
            assert isinstance(stage_mapping[current_running_stage], cls)
            assert stage_mapping[current_running_stage]

        def on_train_dataloader(self) -> None:
            current_running_stage = RunningStage.TRAINING
            self.on_train_dataloader_called = True
            collate_fn = self.train_dataloader().collate_fn  # noqa F811
            assert collate_fn == default_collate
            assert not isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            super().on_train_dataloader()
            collate_fn = self.train_dataloader().collate_fn  # noqa F811
            assert collate_fn.stage == current_running_stage
            self._compare_pre_processor(
                collate_fn, self.data_pipeline.worker_input_transform_processor(current_running_stage)
            )
            assert isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            self._assert_stage_orchestrator_state(self.transfer_batch_to_device._stage_mapping, current_running_stage)

        def on_val_dataloader(self) -> None:
            current_running_stage = RunningStage.VALIDATING
            self.on_val_dataloader_called = True
            collate_fn = self.val_dataloader().collate_fn  # noqa F811
            assert collate_fn == default_collate
            assert isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            super().on_val_dataloader()
            collate_fn = self.val_dataloader().collate_fn  # noqa F811
            assert collate_fn.stage == current_running_stage
            self._compare_pre_processor(
                collate_fn, self.data_pipeline.worker_input_transform_processor(current_running_stage)
            )
            assert isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            self._assert_stage_orchestrator_state(self.transfer_batch_to_device._stage_mapping, current_running_stage)

        def on_test_dataloader(self) -> None:
            current_running_stage = RunningStage.TESTING
            self.on_test_dataloader_called = True
            collate_fn = self.test_dataloader().collate_fn  # noqa F811
            assert collate_fn == default_collate
            assert not isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            super().on_test_dataloader()
            collate_fn = self.test_dataloader().collate_fn  # noqa F811
            assert collate_fn.stage == current_running_stage
            self._compare_pre_processor(
                collate_fn, self.data_pipeline.worker_input_transform_processor(current_running_stage)
            )
            assert isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            self._assert_stage_orchestrator_state(self.transfer_batch_to_device._stage_mapping, current_running_stage)

        def on_predict_dataloader(self) -> None:
            current_running_stage = RunningStage.PREDICTING
            self.on_predict_dataloader_called = True
            collate_fn = self.predict_dataloader().collate_fn  # noqa F811
            assert collate_fn == default_collate
            assert isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            assert self.predict_step == self._saved_predict_step
            super().on_predict_dataloader()
            collate_fn = self.predict_dataloader().collate_fn  # noqa F811
            assert collate_fn.stage == current_running_stage
            self._compare_pre_processor(
                collate_fn, self.data_pipeline.worker_input_transform_processor(current_running_stage)
            )
            assert isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            assert isinstance(self.predict_step, _StageOrchestrator)
            self._assert_stage_orchestrator_state(self.transfer_batch_to_device._stage_mapping, current_running_stage)
            self._assert_stage_orchestrator_state(
                self.predict_step._stage_mapping, current_running_stage, cls=_OutputTransformProcessor
            )

        def on_fit_end(self) -> None:
            super().on_fit_end()
            assert self.train_dataloader().collate_fn == default_collate
            assert self.val_dataloader().collate_fn == default_collate
            assert self.test_dataloader().collate_fn == default_collate
            assert self.predict_dataloader().collate_fn == default_collate
            assert not isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            assert self.predict_step == self._saved_predict_step

    model = TestModel()
    model.data_pipeline = data_pipeline
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model)

    assert model.on_train_dataloader_called
    assert model.on_val_dataloader_called
    assert model.on_test_dataloader_called
    assert model.on_predict_dataloader_called


def test_stage_orchestrator_state_attach_detach(tmpdir):

    model = CustomModel()
    input_transform = TestInputTransform()

    _original_predict_step = model.predict_step

    class CustomDataPipeline(DataPipeline):
        def _attach_output_transform_to_model(
            self, model: "Task", _output_transform_processor: _OutputTransformProcessor
        ) -> "Task":
            model.predict_step = self._model_predict_step_wrapper(
                model.predict_step, _output_transform_processor, model
            )
            return model

    data_pipeline = CustomDataPipeline(input_transform=input_transform)
    _output_transform_processor = data_pipeline._create_output_transform_processor(RunningStage.PREDICTING)
    data_pipeline._attach_output_transform_to_model(model, _output_transform_processor)
    assert model.predict_step._original == _original_predict_step
    assert model.predict_step._stage_mapping[RunningStage.PREDICTING] == _output_transform_processor
    data_pipeline._detach_output_transform_from_model(model)
    assert model.predict_step == _original_predict_step


class LamdaDummyDataset(torch.utils.data.Dataset):
    def __init__(self, fx: Callable):
        self.fx = fx

    def __getitem__(self, index: int) -> Any:
        return self.fx()

    def __len__(self) -> int:
        return 5


class TestInputTransformationsInput(Input):
    def __init__(self):
        super().__init__()

        self.train_load_data_called = False
        self.val_load_data_called = False
        self.val_load_sample_called = False
        self.test_load_data_called = False
        self.predict_load_data_called = False

    @staticmethod
    def fn_train_load_data() -> Tuple:
        return (
            0,
            1,
            2,
            3,
        )

    def train_load_data(self, sample) -> LamdaDummyDataset:
        assert self.training
        assert self.current_fn == "load_data"
        self.train_load_data_called = True
        return LamdaDummyDataset(self.fn_train_load_data)

    def val_load_data(self, sample, dataset) -> List[int]:
        assert self.validating
        assert self.current_fn == "load_data"
        self.val_load_data_called = True
        return list(range(5))

    def val_load_sample(self, sample) -> Dict[str, Tensor]:
        assert self.validating
        assert self.current_fn == "load_sample"
        self.val_load_sample_called = True
        return {"a": sample, "b": sample + 1}

    @staticmethod
    def fn_test_load_data() -> List[torch.Tensor]:
        return [torch.rand(1), torch.rand(1)]

    def test_load_data(self, sample) -> LamdaDummyDataset:
        assert self.testing
        assert self.current_fn == "load_data"
        self.test_load_data_called = True
        return LamdaDummyDataset(self.fn_test_load_data)

    @staticmethod
    def fn_predict_load_data() -> List[str]:
        return ["a", "b"]

    def predict_load_data(self, sample) -> LamdaDummyDataset:
        assert self.predicting
        assert self.current_fn == "load_data"
        self.predict_load_data_called = True
        return LamdaDummyDataset(self.fn_predict_load_data)


class TestInputTransformations(DefaultInputTransform):
    def __init__(self):
        super().__init__(data_sources={"default": TestInputTransformationsInput()})

        self.train_pre_tensor_transform_called = False
        self.train_collate_called = False
        self.train_per_batch_transform_on_device_called = False
        self.val_to_tensor_transform_called = False
        self.val_collate_called = False
        self.val_per_batch_transform_on_device_called = False
        self.test_to_tensor_transform_called = False
        self.test_post_tensor_transform_called = False

    def train_pre_tensor_transform(self, sample: Any) -> Any:
        assert self.training
        assert self.current_fn == "pre_tensor_transform"
        self.train_pre_tensor_transform_called = True
        return sample + (5,)

    def train_collate(self, samples) -> Tensor:
        assert self.training
        assert self.current_fn == "collate"
        self.train_collate_called = True
        return tensor([list(s) for s in samples])

    def train_per_batch_transform_on_device(self, batch: Any) -> Any:
        assert self.training
        assert self.current_fn == "per_batch_transform_on_device"
        self.train_per_batch_transform_on_device_called = True
        assert torch.equal(batch, tensor([[0, 1, 2, 3, 5], [0, 1, 2, 3, 5]]))

    def val_to_tensor_transform(self, sample: Any) -> Tensor:
        assert self.validating
        assert self.current_fn == "to_tensor_transform"
        self.val_to_tensor_transform_called = True
        return sample

    def val_collate(self, samples) -> Dict[str, Tensor]:
        assert self.validating
        assert self.current_fn == "collate"
        self.val_collate_called = True
        _count = samples[0]["a"]
        assert samples == [{"a": _count, "b": _count + 1}, {"a": _count + 1, "b": _count + 2}]
        return {"a": tensor([0, 1]), "b": tensor([1, 2])}

    def val_per_batch_transform_on_device(self, batch: Any) -> Any:
        assert self.validating
        assert self.current_fn == "per_batch_transform_on_device"
        self.val_per_batch_transform_on_device_called = True
        if isinstance(batch, list):
            batch = batch[0]
        assert torch.equal(batch["a"], tensor([0, 1]))
        assert torch.equal(batch["b"], tensor([1, 2]))
        return [False]

    def test_to_tensor_transform(self, sample: Any) -> Tensor:
        assert self.testing
        assert self.current_fn == "to_tensor_transform"
        self.test_to_tensor_transform_called = True
        return sample

    def test_post_tensor_transform(self, sample: Tensor) -> Tensor:
        assert self.testing
        assert self.current_fn == "post_tensor_transform"
        self.test_post_tensor_transform_called = True
        return sample


class TestInputTransformations2(TestInputTransformations):
    def val_to_tensor_transform(self, sample: Any) -> Tensor:
        self.val_to_tensor_transform_called = True
        return {"a": tensor(sample["a"]), "b": tensor(sample["b"])}


class CustomModel(Task):
    def __init__(self):
        super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())

    def training_step(self, batch, batch_idx):
        assert batch is None

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        assert batch is False

    def test_step(self, batch, batch_idx):
        assert len(batch) == 2
        assert batch[0].shape == torch.Size([2, 1])

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        assert batch[0][0] == "a"
        assert batch[0][1] == "a"
        assert batch[1][0] == "b"
        assert batch[1][1] == "b"
        return tensor([0, 0, 0])


def test_datapipeline_transformations(tmpdir):

    datamodule = DataModule.from_data_source(
        "default", 1, 1, 1, 1, batch_size=2, num_workers=0, input_transform=TestInputTransformations()
    )

    assert datamodule.train_dataloader().dataset[0] == (0, 1, 2, 3)
    batch = next(iter(datamodule.train_dataloader()))
    assert torch.equal(batch, tensor([[0, 1, 2, 3, 5], [0, 1, 2, 3, 5]]))

    assert datamodule.val_dataloader().dataset[0] == {"a": 0, "b": 1}
    assert datamodule.val_dataloader().dataset[1] == {"a": 1, "b": 2}
    with pytest.raises(MisconfigurationException, match="When ``to_tensor_transform``"):
        batch = next(iter(datamodule.val_dataloader()))

    datamodule = DataModule.from_data_source(
        "default", 1, 1, 1, 1, batch_size=2, num_workers=0, input_transform=TestInputTransformations2()
    )
    batch = next(iter(datamodule.val_dataloader()))
    assert torch.equal(batch["a"], tensor([0, 1]))
    assert torch.equal(batch["b"], tensor([1, 2]))

    model = CustomModel()
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        limit_test_batches=2,
        limit_predict_batches=2,
        num_sanity_val_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model)
    trainer.predict(model)

    input_transform = model._input_transform
    data_source = input_transform.data_source_of_name("default")
    assert data_source.train_load_data_called
    assert input_transform.train_pre_tensor_transform_called
    assert input_transform.train_collate_called
    assert input_transform.train_per_batch_transform_on_device_called
    assert data_source.val_load_data_called
    assert data_source.val_load_sample_called
    assert input_transform.val_to_tensor_transform_called
    assert input_transform.val_collate_called
    assert input_transform.val_per_batch_transform_on_device_called
    assert data_source.test_load_data_called
    assert input_transform.test_to_tensor_transform_called
    assert input_transform.test_post_tensor_transform_called
    assert data_source.predict_load_data_called


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_datapipeline_transformations_overridden_by_task():
    # define input transforms
    class ImageInput(Input):
        def load_data(self, folder: str):
            # from folder -> return files paths
            return ["a.jpg", "b.jpg"]

        def load_sample(self, path: str) -> Image.Image:
            # from a file path, load the associated image
            return np.random.uniform(0, 1, (64, 64, 3))

    class ImageClassificationInputTransform(DefaultInputTransform):
        def __init__(
            self,
            train_transform=None,
            val_transform=None,
            test_transform=None,
            predict_transform=None,
        ):
            super().__init__(
                train_transform=train_transform,
                val_transform=val_transform,
                test_transform=test_transform,
                predict_transform=predict_transform,
                data_sources={"default": ImageInput()},
            )

        def default_transforms(self):
            return {
                "to_tensor_transform": T.Compose([T.ToTensor()]),
                "per_batch_transform_on_device": T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            }

    # define task which overrides transforms using set_state
    class CustomModel(Task):
        def __init__(self):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())

            # override default transform to resize images
            self.set_state(ToTensorTransform(T.Compose([T.ToTensor(), T.Resize(128)])))

            # remove normalization, => image still in [0, 1] range
            self.set_state(PerBatchTransformOnDevice(None))

        def training_step(self, batch, batch_idx):
            assert batch.shape == torch.Size([2, 3, 128, 128])
            assert torch.max(batch) <= 1.0
            assert torch.min(batch) >= 0.0

        def validation_step(self, batch, batch_idx):
            assert batch.shape == torch.Size([2, 3, 128, 128])
            assert torch.max(batch) <= 1.0
            assert torch.min(batch) >= 0.0

    class CustomDataModule(DataModule):

        input_transform_cls = ImageClassificationInputTransform

    datamodule = CustomDataModule.from_data_source(
        "default",
        "train_folder",
        "val_folder",
        None,
        batch_size=2,
    )

    # call trainer
    model = CustomModel()
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)


def test_is_overriden_recursive(tmpdir):
    class TestInputTransform(DefaultInputTransform):
        def collate(self, *_):
            pass

        def val_collate(self, *_):
            pass

    input_transform = TestInputTransform()
    assert DataPipeline._is_overriden_recursive("collate", input_transform, InputTransform, prefix="val")
    assert DataPipeline._is_overriden_recursive("collate", input_transform, InputTransform, prefix="train")
    assert not DataPipeline._is_overriden_recursive(
        "per_batch_transform_on_device", input_transform, InputTransform, prefix="train"
    )
    assert not DataPipeline._is_overriden_recursive("per_batch_transform_on_device", input_transform, InputTransform)
    with pytest.raises(MisconfigurationException, match="This function doesn't belong to the parent class"):
        assert not DataPipeline._is_overriden_recursive("chocolate", input_transform, InputTransform)


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
@patch("torch.save")  # need to mock torch.save or we get pickle error
def test_dummy_example(tmpdir):
    class ImageInput(Input):
        def load_data(self, folder: str):
            # from folder -> return files paths
            return ["a.jpg", "b.jpg"]

        def load_sample(self, path: str) -> Image.Image:
            # from a file path, load the associated image
            img8Bit = np.uint8(np.random.uniform(0, 1, (64, 64, 3)) * 255.0)
            return Image.fromarray(img8Bit)

    class ImageClassificationInputTransform(DefaultInputTransform):
        def __init__(
            self,
            train_transform=None,
            val_transform=None,
            test_transform=None,
            predict_transform=None,
            to_tensor_transform=None,
            train_per_sample_transform_on_device=None,
        ):
            super().__init__(
                train_transform=train_transform,
                val_transform=val_transform,
                test_transform=test_transform,
                predict_transform=predict_transform,
                data_sources={"default": ImageInput()},
            )
            self._to_tensor = to_tensor_transform
            self._train_per_sample_transform_on_device = train_per_sample_transform_on_device

        def to_tensor_transform(self, pil_image: Image.Image) -> Tensor:
            # convert pil image into a tensor
            return self._to_tensor(pil_image)

        def train_per_sample_transform_on_device(self, sample: Any) -> Any:
            # apply an augmentation per sample on gpu for train only
            return self._train_per_sample_transform_on_device(sample)

    class CustomModel(Task):
        def __init__(self):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())

        def training_step(self, batch, batch_idx):
            assert batch.shape == torch.Size([2, 3, 64, 64])

        def validation_step(self, batch, batch_idx):
            assert batch.shape == torch.Size([2, 3, 64, 64])

        def test_step(self, batch, batch_idx):
            assert batch.shape == torch.Size([2, 3, 64, 64])

    class CustomDataModule(DataModule):

        input_transform_cls = ImageClassificationInputTransform

    datamodule = CustomDataModule.from_data_source(
        "default",
        "train_folder",
        "val_folder",
        "test_folder",
        None,
        batch_size=2,
        to_tensor_transform=T.ToTensor(),
        train_per_sample_transform_on_device=T.RandomHorizontalFlip(),
    )

    assert isinstance(datamodule.train_dataloader().dataset[0], Image.Image)
    batch = next(iter(datamodule.train_dataloader()))
    assert batch[0].shape == torch.Size([3, 64, 64])

    model = CustomModel()
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        limit_test_batches=2,
        limit_predict_batches=2,
        num_sanity_val_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model)


def test_input_transform_transforms(tmpdir):
    """This test makes sure that when a input_transform is being provided transforms as dictionaries, checking is
    done properly, and collate_in_worker_from_transform is properly extracted."""

    with pytest.raises(MisconfigurationException, match="Transform should be a dict."):
        DefaultInputTransform(train_transform="choco")

    with pytest.raises(MisconfigurationException, match="train_transform contains {'choco'}. Only"):
        DefaultInputTransform(train_transform={"choco": None})

    input_transform = DefaultInputTransform(train_transform={"to_tensor_transform": torch.nn.Linear(1, 1)})
    # keep is None
    assert input_transform._train_collate_in_worker_from_transform is True
    assert input_transform._val_collate_in_worker_from_transform is None
    assert input_transform._test_collate_in_worker_from_transform is None
    assert input_transform._predict_collate_in_worker_from_transform is None

    with pytest.raises(MisconfigurationException, match="`per_batch_transform` and `per_sample_transform_on_device`"):
        input_transform = DefaultInputTransform(
            train_transform={
                "per_batch_transform": torch.nn.Linear(1, 1),
                "per_sample_transform_on_device": torch.nn.Linear(1, 1),
            }
        )

    input_transform = DefaultInputTransform(
        train_transform={"per_batch_transform": torch.nn.Linear(1, 1)},
        predict_transform={"per_sample_transform_on_device": torch.nn.Linear(1, 1)},
    )
    # keep is None
    assert input_transform._train_collate_in_worker_from_transform is True
    assert input_transform._val_collate_in_worker_from_transform is None
    assert input_transform._test_collate_in_worker_from_transform is None
    assert input_transform._predict_collate_in_worker_from_transform is False

    train_input_transform_processor = DataPipeline(input_transform=input_transform).worker_input_transform_processor(
        RunningStage.TRAINING
    )
    val_input_transform_processor = DataPipeline(input_transform=input_transform).worker_input_transform_processor(
        RunningStage.VALIDATING
    )
    test_input_transform_processor = DataPipeline(input_transform=input_transform).worker_input_transform_processor(
        RunningStage.TESTING
    )
    predict_input_transform_processor = DataPipeline(input_transform=input_transform).worker_input_transform_processor(
        RunningStage.PREDICTING
    )

    assert train_input_transform_processor.collate_fn.func == input_transform.collate
    assert val_input_transform_processor.collate_fn.func == input_transform.collate
    assert test_input_transform_processor.collate_fn.func == input_transform.collate
    assert predict_input_transform_processor.collate_fn.func == DataPipeline._identity

    class CustomInputTransform(DefaultInputTransform):
        def per_sample_transform_on_device(self, sample: Any) -> Any:
            return super().per_sample_transform_on_device(sample)

        def per_batch_transform(self, batch: Any) -> Any:
            return super().per_batch_transform(batch)

    input_transform = CustomInputTransform(
        train_transform={"per_batch_transform": torch.nn.Linear(1, 1)},
        predict_transform={"per_sample_transform_on_device": torch.nn.Linear(1, 1)},
    )
    # keep is None
    assert input_transform._train_collate_in_worker_from_transform is True
    assert input_transform._val_collate_in_worker_from_transform is None
    assert input_transform._test_collate_in_worker_from_transform is None
    assert input_transform._predict_collate_in_worker_from_transform is False

    data_pipeline = DataPipeline(input_transform=input_transform)

    train_input_transform_processor = data_pipeline.worker_input_transform_processor(RunningStage.TRAINING)
    with pytest.raises(MisconfigurationException, match="`per_batch_transform` and `per_sample_transform_on_device`"):
        val_input_transform_processor = data_pipeline.worker_input_transform_processor(RunningStage.VALIDATING)
    with pytest.raises(MisconfigurationException, match="`per_batch_transform` and `per_sample_transform_on_device`"):
        test_input_transform_processor = data_pipeline.worker_input_transform_processor(RunningStage.TESTING)
    predict_input_transform_processor = data_pipeline.worker_input_transform_processor(RunningStage.PREDICTING)

    assert train_input_transform_processor.collate_fn.func == input_transform.collate
    assert predict_input_transform_processor.collate_fn.func == DataPipeline._identity


def test_iterable_auto_dataset(tmpdir):
    class CustomInput(Input):
        def load_sample(self, index: int) -> Dict[str, int]:
            return {"index": index}

    ds = IterableAutoDataset(range(10), data_source=CustomInput(), running_stage=RunningStage.TRAINING)

    for index, v in enumerate(ds):
        assert v == {"index": index}


class CustomInputTransformHyperparameters(DefaultInputTransform):
    def __init__(self, token: str, *args, **kwargs):
        self.token = token
        super().__init__(*args, **kwargs)

    @classmethod
    def load_from_state_dict(cls, state_dict: Dict[str, Any]):
        return cls(state_dict["token"])

    def state_dict(self) -> Dict[str, Any]:
        return {"token": self.token}


def local_fn(x):
    return x


def test_save_hyperparemeters(tmpdir):

    kwargs = {"train_transform": {"pre_tensor_transform": local_fn}}
    input_transform = CustomInputTransformHyperparameters("token", **kwargs)
    state_dict = input_transform.state_dict()
    torch.save(state_dict, os.path.join(tmpdir, "state_dict.pt"))
    state_dict = torch.load(os.path.join(tmpdir, "state_dict.pt"))
    input_transform = CustomInputTransformHyperparameters.load_from_state_dict(state_dict)
    assert isinstance(input_transform, CustomInputTransformHyperparameters)

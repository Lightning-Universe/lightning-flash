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
from typing import Any, cast, Dict, Optional, Tuple

import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from flash import Trainer
from flash.core.data.data_pipeline import _StageOrchestrator, DataPipeline, DataPipelineState
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import _InputTransformProcessor, DefaultInputTransform, InputTransform
from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import _OutputTransformProcessor, OutputTransform
from flash.core.data.process import Deserializer
from flash.core.data.properties import ProcessState
from flash.core.model import Task
from flash.core.utilities.stages import RunningStage


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
        input=cast(Input, "input"),
        input_transform=cast(InputTransform, "input_transform"),
        output_transform=cast(OutputTransform, "output_transform"),
        output=cast(Output, "output"),
        deserializer=cast(Deserializer, "deserializer"),
    )

    expected = "input=input, deserializer=deserializer, "
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


def test_data_pipeline_is_overridden_and_resolve_function_hierarchy(tmpdir):
    class CustomInputTransform(DefaultInputTransform):
        def val_per_sample_transform(self, *_, **__):
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

    # per_sample_transform
    assert train_func_names["per_sample_transform"] == "per_sample_transform"
    assert val_func_names["per_sample_transform"] == "val_per_sample_transform"
    assert test_func_names["per_sample_transform"] == "per_sample_transform"
    assert predict_func_names["per_sample_transform"] == "per_sample_transform"

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

    assert train_worker_input_transform_processor.per_sample_transform.func == input_transform.per_sample_transform
    assert train_worker_input_transform_processor.collate_fn.func == input_transform.collate
    assert train_worker_input_transform_processor.per_batch_transform.func == input_transform.per_batch_transform

    assert val_worker_input_transform_processor.per_sample_transform.func == input_transform.val_per_sample_transform
    assert val_worker_input_transform_processor.collate_fn.func == DataPipeline._identity
    assert val_worker_input_transform_processor.per_batch_transform.func == input_transform.per_batch_transform

    assert test_worker_input_transform_processor.per_sample_transform.func == input_transform.per_sample_transform
    assert test_worker_input_transform_processor.collate_fn.func == input_transform.test_collate
    assert test_worker_input_transform_processor.per_batch_transform.func == input_transform.per_batch_transform

    assert predict_worker_input_transform_processor.per_sample_transform.func == input_transform.per_sample_transform
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
            self._saved_predict_step = self.predict_step

        @staticmethod
        def _compare_pre_processor(p1, p2):
            assert p1.per_sample_transform.func == p2.per_sample_transform.func
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


def test_is_overridden_recursive(tmpdir):
    class TestInputTransform(DefaultInputTransform):
        def collate(self, *_):
            pass

        def val_collate(self, *_):
            pass

    input_transform = TestInputTransform()
    assert DataPipeline._is_overridden_recursive("collate", input_transform, InputTransform, prefix="val")
    assert DataPipeline._is_overridden_recursive("collate", input_transform, InputTransform, prefix="train")
    assert not DataPipeline._is_overridden_recursive(
        "per_batch_transform_on_device", input_transform, InputTransform, prefix="train"
    )
    assert not DataPipeline._is_overridden_recursive("per_batch_transform_on_device", input_transform, InputTransform)
    with pytest.raises(MisconfigurationException, match="This function doesn't belong to the parent class"):
        assert not DataPipeline._is_overridden_recursive("chocolate", input_transform, InputTransform)


def test_input_transform_transforms(tmpdir):
    """This test makes sure that when a input_transform is being provided transforms as dictionaries, checking is
    done properly, and collate_in_worker_from_transform is properly extracted."""

    with pytest.raises(MisconfigurationException, match="Transform should be a dict."):
        DefaultInputTransform(train_transform="choco")

    with pytest.raises(MisconfigurationException, match="train_transform contains {'choco'}. Only"):
        DefaultInputTransform(train_transform={"choco": None})

    input_transform = DefaultInputTransform(train_transform={"per_sample_transform": torch.nn.Linear(1, 1)})
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

    kwargs = {"train_transform": {"per_sample_transform": local_fn}}
    input_transform = CustomInputTransformHyperparameters("token", **kwargs)
    state_dict = input_transform.state_dict()
    torch.save(state_dict, os.path.join(tmpdir, "state_dict.pt"))
    state_dict = torch.load(os.path.join(tmpdir, "state_dict.pt"))
    input_transform = CustomInputTransformHyperparameters.load_from_state_dict(state_dict)
    assert isinstance(input_transform, CustomInputTransformHyperparameters)

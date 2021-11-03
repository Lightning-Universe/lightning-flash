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
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from flash.core.data.auto_dataset import IterableAutoDataset
from flash.core.data.batch import _Postprocessor, _Preprocessor
from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import _StageOrchestrator, DataPipeline, DataPipelineState
from flash.core.data.io.input import BaseInput
from flash.core.data.process import DefaultPreprocess, Deserializer, Postprocess, Preprocess, Serializer
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
        data_source=cast(BaseInput, "data_source"),
        preprocess=cast(Preprocess, "preprocess"),
        postprocess=cast(Postprocess, "postprocess"),
        serializer=cast(Serializer, "serializer"),
        deserializer=cast(Deserializer, "deserializer"),
    )

    expected = "data_source=data_source, deserializer=deserializer, "
    expected += "preprocess=preprocess, postprocess=postprocess, serializer=serializer"
    assert str(data_pipeline) == (f"DataPipeline({expected})")


@pytest.mark.parametrize("use_preprocess", [False, True])
@pytest.mark.parametrize("use_postprocess", [False, True])
def test_data_pipeline_init_and_assignement(use_preprocess, use_postprocess, tmpdir):
    class CustomModel(Task):
        def __init__(self, postprocess: Optional[Postprocess] = None):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())
            self._postprocess = postprocess

        def train_dataloader(self) -> Any:
            return DataLoader(DummyDataset())

    class SubPreprocess(DefaultPreprocess):
        pass

    class SubPostprocess(Postprocess):
        pass

    data_pipeline = DataPipeline(
        preprocess=SubPreprocess() if use_preprocess else None,
        postprocess=SubPostprocess() if use_postprocess else None,
    )
    assert isinstance(data_pipeline._preprocess_pipeline, SubPreprocess if use_preprocess else DefaultPreprocess)
    assert isinstance(data_pipeline._postprocess_pipeline, SubPostprocess if use_postprocess else Postprocess)

    model = CustomModel(postprocess=Postprocess())
    model.data_pipeline = data_pipeline
    # TODO: the line below should make the same effect but it's not
    # data_pipeline._attach_to_model(model)

    if use_preprocess:
        assert isinstance(model._preprocess, SubPreprocess)
    else:
        assert model._preprocess is None or isinstance(model._preprocess, Preprocess)

    if use_postprocess:
        assert isinstance(model._postprocess, SubPostprocess)
    else:
        assert model._postprocess is None or isinstance(model._postprocess, Postprocess)


def test_data_pipeline_is_overriden_and_resolve_function_hierarchy(tmpdir):
    class CustomPreprocess(DefaultPreprocess):
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

    preprocess = CustomPreprocess()
    data_pipeline = DataPipeline(preprocess=preprocess)

    train_func_names: Dict[str, str] = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._preprocess_pipeline, RunningStage.TRAINING, Preprocess
        )
        for k in data_pipeline.PREPROCESS_FUNCS
    }
    val_func_names: Dict[str, str] = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._preprocess_pipeline, RunningStage.VALIDATING, Preprocess
        )
        for k in data_pipeline.PREPROCESS_FUNCS
    }
    test_func_names: Dict[str, str] = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._preprocess_pipeline, RunningStage.TESTING, Preprocess
        )
        for k in data_pipeline.PREPROCESS_FUNCS
    }
    predict_func_names: Dict[str, str] = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._preprocess_pipeline, RunningStage.PREDICTING, Preprocess
        )
        for k in data_pipeline.PREPROCESS_FUNCS
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

    train_worker_preprocessor = data_pipeline.worker_preprocessor(RunningStage.TRAINING)
    val_worker_preprocessor = data_pipeline.worker_preprocessor(RunningStage.VALIDATING)
    test_worker_preprocessor = data_pipeline.worker_preprocessor(RunningStage.TESTING)
    predict_worker_preprocessor = data_pipeline.worker_preprocessor(RunningStage.PREDICTING)

    _seq = train_worker_preprocessor.per_sample_transform
    assert _seq.pre_tensor_transform.func == preprocess.pre_tensor_transform
    assert _seq.to_tensor_transform.func == preprocess.to_tensor_transform
    assert _seq.post_tensor_transform.func == preprocess.train_post_tensor_transform
    assert train_worker_preprocessor.collate_fn.func == preprocess.collate
    assert train_worker_preprocessor.per_batch_transform.func == preprocess.per_batch_transform

    _seq = val_worker_preprocessor.per_sample_transform
    assert _seq.pre_tensor_transform.func == preprocess.val_pre_tensor_transform
    assert _seq.to_tensor_transform.func == preprocess.to_tensor_transform
    assert _seq.post_tensor_transform.func == preprocess.post_tensor_transform
    assert val_worker_preprocessor.collate_fn.func == DataPipeline._identity
    assert val_worker_preprocessor.per_batch_transform.func == preprocess.per_batch_transform

    _seq = test_worker_preprocessor.per_sample_transform
    assert _seq.pre_tensor_transform.func == preprocess.pre_tensor_transform
    assert _seq.to_tensor_transform.func == preprocess.to_tensor_transform
    assert _seq.post_tensor_transform.func == preprocess.post_tensor_transform
    assert test_worker_preprocessor.collate_fn.func == preprocess.test_collate
    assert test_worker_preprocessor.per_batch_transform.func == preprocess.per_batch_transform

    _seq = predict_worker_preprocessor.per_sample_transform
    assert _seq.pre_tensor_transform.func == preprocess.pre_tensor_transform
    assert _seq.to_tensor_transform.func == preprocess.predict_to_tensor_transform
    assert _seq.post_tensor_transform.func == preprocess.post_tensor_transform
    assert predict_worker_preprocessor.collate_fn.func == preprocess.collate
    assert predict_worker_preprocessor.per_batch_transform.func == preprocess.per_batch_transform


class CustomPreprocess(DefaultPreprocess):
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


def test_data_pipeline_predict_worker_preprocessor_and_device_preprocessor():

    preprocess = CustomPreprocess()
    data_pipeline = DataPipeline(preprocess=preprocess)

    data_pipeline.worker_preprocessor(RunningStage.TRAINING)
    with pytest.raises(MisconfigurationException, match="are mutually exclusive"):
        data_pipeline.worker_preprocessor(RunningStage.VALIDATING)
    with pytest.raises(MisconfigurationException, match="are mutually exclusive"):
        data_pipeline.worker_preprocessor(RunningStage.TESTING)
    data_pipeline.worker_preprocessor(RunningStage.PREDICTING)


def test_detach_preprocessing_from_model(tmpdir):
    class CustomModel(Task):
        def __init__(self, postprocess: Optional[Postprocess] = None):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())
            self._postprocess = postprocess

        def train_dataloader(self) -> Any:
            return DataLoader(DummyDataset())

    preprocess = CustomPreprocess()
    data_pipeline = DataPipeline(preprocess=preprocess)
    model = CustomModel()
    model.data_pipeline = data_pipeline

    assert model.train_dataloader().collate_fn == default_collate
    assert model.transfer_batch_to_device.__self__ == model
    model.on_train_dataloader()
    assert isinstance(model.train_dataloader().collate_fn, _Preprocessor)
    assert isinstance(model.transfer_batch_to_device, _StageOrchestrator)
    model.on_fit_end()
    assert model.transfer_batch_to_device.__self__ == model
    assert model.train_dataloader().collate_fn == default_collate


class TestPreprocess(DefaultPreprocess):
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
    class SubPreprocess(DefaultPreprocess):
        pass

    preprocess = SubPreprocess()
    data_pipeline = DataPipeline(preprocess=preprocess)

    class CustomModel(Task):
        def __init__(self):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())
            self._postprocess = Postprocess()

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
            stage_mapping: Dict, current_running_stage: RunningStage, cls=_Preprocessor
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
            self._compare_pre_processor(collate_fn, self.data_pipeline.worker_preprocessor(current_running_stage))
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
            self._compare_pre_processor(collate_fn, self.data_pipeline.worker_preprocessor(current_running_stage))
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
            self._compare_pre_processor(collate_fn, self.data_pipeline.worker_preprocessor(current_running_stage))
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
            self._compare_pre_processor(collate_fn, self.data_pipeline.worker_preprocessor(current_running_stage))
            assert isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            assert isinstance(self.predict_step, _StageOrchestrator)
            self._assert_stage_orchestrator_state(self.transfer_batch_to_device._stage_mapping, current_running_stage)
            self._assert_stage_orchestrator_state(
                self.predict_step._stage_mapping, current_running_stage, cls=_Postprocessor
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
    preprocess = TestPreprocess()

    _original_predict_step = model.predict_step

    class CustomDataPipeline(DataPipeline):
        def _attach_postprocess_to_model(self, model: "Task", _postprocesssor: _Postprocessor) -> "Task":
            model.predict_step = self._model_predict_step_wrapper(model.predict_step, _postprocesssor, model)
            return model

    data_pipeline = CustomDataPipeline(preprocess=preprocess)
    _postprocesssor = data_pipeline._create_uncollate_postprocessors(RunningStage.PREDICTING)
    data_pipeline._attach_postprocess_to_model(model, _postprocesssor)
    assert model.predict_step._original == _original_predict_step
    assert model.predict_step._stage_mapping[RunningStage.PREDICTING] == _postprocesssor
    data_pipeline._detach_postprocess_from_model(model)
    assert model.predict_step == _original_predict_step


class LamdaDummyDataset(torch.utils.data.Dataset):
    def __init__(self, fx: Callable):
        self.fx = fx

    def __getitem__(self, index: int) -> Any:
        return self.fx()

    def __len__(self) -> int:
        return 5


class TestInputTransformationsInput(BaseInput):
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


class TestInputTransformations(DefaultPreprocess):
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
        "default", 1, 1, 1, 1, batch_size=2, num_workers=0, preprocess=TestInputTransformations()
    )

    assert datamodule.train_dataloader().dataset[0] == (0, 1, 2, 3)
    batch = next(iter(datamodule.train_dataloader()))
    assert torch.equal(batch, tensor([[0, 1, 2, 3, 5], [0, 1, 2, 3, 5]]))

    assert datamodule.val_dataloader().dataset[0] == {"a": 0, "b": 1}
    assert datamodule.val_dataloader().dataset[1] == {"a": 1, "b": 2}
    with pytest.raises(MisconfigurationException, match="When ``to_tensor_transform``"):
        batch = next(iter(datamodule.val_dataloader()))

    datamodule = DataModule.from_data_source(
        "default", 1, 1, 1, 1, batch_size=2, num_workers=0, preprocess=TestInputTransformations2()
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

    preprocess = model._preprocess
    data_source = preprocess.data_source_of_name("default")
    assert data_source.train_load_data_called
    assert preprocess.train_pre_tensor_transform_called
    assert preprocess.train_collate_called
    assert preprocess.train_per_batch_transform_on_device_called
    assert data_source.val_load_data_called
    assert data_source.val_load_sample_called
    assert preprocess.val_to_tensor_transform_called
    assert preprocess.val_collate_called
    assert preprocess.val_per_batch_transform_on_device_called
    assert data_source.test_load_data_called
    assert preprocess.test_to_tensor_transform_called
    assert preprocess.test_post_tensor_transform_called
    assert data_source.predict_load_data_called


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_datapipeline_transformations_overridden_by_task():
    # define preprocess transforms
    class ImageInput(BaseInput):
        def load_data(self, folder: str):
            # from folder -> return files paths
            return ["a.jpg", "b.jpg"]

        def load_sample(self, path: str) -> Image.Image:
            # from a file path, load the associated image
            return np.random.uniform(0, 1, (64, 64, 3))

    class ImageClassificationPreprocess(DefaultPreprocess):
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

        preprocess_cls = ImageClassificationPreprocess

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
    class TestPreprocess(DefaultPreprocess):
        def collate(self, *_):
            pass

        def val_collate(self, *_):
            pass

    preprocess = TestPreprocess()
    assert DataPipeline._is_overriden_recursive("collate", preprocess, Preprocess, prefix="val")
    assert DataPipeline._is_overriden_recursive("collate", preprocess, Preprocess, prefix="train")
    assert not DataPipeline._is_overriden_recursive(
        "per_batch_transform_on_device", preprocess, Preprocess, prefix="train"
    )
    assert not DataPipeline._is_overriden_recursive("per_batch_transform_on_device", preprocess, Preprocess)
    with pytest.raises(MisconfigurationException, match="This function doesn't belong to the parent class"):
        assert not DataPipeline._is_overriden_recursive("chocolate", preprocess, Preprocess)


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
@patch("torch.save")  # need to mock torch.save or we get pickle error
def test_dummy_example(tmpdir):
    class ImageInput(BaseInput):
        def load_data(self, folder: str):
            # from folder -> return files paths
            return ["a.jpg", "b.jpg"]

        def load_sample(self, path: str) -> Image.Image:
            # from a file path, load the associated image
            img8Bit = np.uint8(np.random.uniform(0, 1, (64, 64, 3)) * 255.0)
            return Image.fromarray(img8Bit)

    class ImageClassificationPreprocess(DefaultPreprocess):
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

        preprocess_cls = ImageClassificationPreprocess

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


def test_preprocess_transforms(tmpdir):
    """This test makes sure that when a preprocess is being provided transforms as dictionaries, checking is done
    properly, and collate_in_worker_from_transform is properly extracted."""

    with pytest.raises(MisconfigurationException, match="Transform should be a dict."):
        DefaultPreprocess(train_transform="choco")

    with pytest.raises(MisconfigurationException, match="train_transform contains {'choco'}. Only"):
        DefaultPreprocess(train_transform={"choco": None})

    preprocess = DefaultPreprocess(train_transform={"to_tensor_transform": torch.nn.Linear(1, 1)})
    # keep is None
    assert preprocess._train_collate_in_worker_from_transform is True
    assert preprocess._val_collate_in_worker_from_transform is None
    assert preprocess._test_collate_in_worker_from_transform is None
    assert preprocess._predict_collate_in_worker_from_transform is None

    with pytest.raises(MisconfigurationException, match="`per_batch_transform` and `per_sample_transform_on_device`"):
        preprocess = DefaultPreprocess(
            train_transform={
                "per_batch_transform": torch.nn.Linear(1, 1),
                "per_sample_transform_on_device": torch.nn.Linear(1, 1),
            }
        )

    preprocess = DefaultPreprocess(
        train_transform={"per_batch_transform": torch.nn.Linear(1, 1)},
        predict_transform={"per_sample_transform_on_device": torch.nn.Linear(1, 1)},
    )
    # keep is None
    assert preprocess._train_collate_in_worker_from_transform is True
    assert preprocess._val_collate_in_worker_from_transform is None
    assert preprocess._test_collate_in_worker_from_transform is None
    assert preprocess._predict_collate_in_worker_from_transform is False

    train_preprocessor = DataPipeline(preprocess=preprocess).worker_preprocessor(RunningStage.TRAINING)
    val_preprocessor = DataPipeline(preprocess=preprocess).worker_preprocessor(RunningStage.VALIDATING)
    test_preprocessor = DataPipeline(preprocess=preprocess).worker_preprocessor(RunningStage.TESTING)
    predict_preprocessor = DataPipeline(preprocess=preprocess).worker_preprocessor(RunningStage.PREDICTING)

    assert train_preprocessor.collate_fn.func == preprocess.collate
    assert val_preprocessor.collate_fn.func == preprocess.collate
    assert test_preprocessor.collate_fn.func == preprocess.collate
    assert predict_preprocessor.collate_fn.func == DataPipeline._identity

    class CustomPreprocess(DefaultPreprocess):
        def per_sample_transform_on_device(self, sample: Any) -> Any:
            return super().per_sample_transform_on_device(sample)

        def per_batch_transform(self, batch: Any) -> Any:
            return super().per_batch_transform(batch)

    preprocess = CustomPreprocess(
        train_transform={"per_batch_transform": torch.nn.Linear(1, 1)},
        predict_transform={"per_sample_transform_on_device": torch.nn.Linear(1, 1)},
    )
    # keep is None
    assert preprocess._train_collate_in_worker_from_transform is True
    assert preprocess._val_collate_in_worker_from_transform is None
    assert preprocess._test_collate_in_worker_from_transform is None
    assert preprocess._predict_collate_in_worker_from_transform is False

    data_pipeline = DataPipeline(preprocess=preprocess)

    train_preprocessor = data_pipeline.worker_preprocessor(RunningStage.TRAINING)
    with pytest.raises(MisconfigurationException, match="`per_batch_transform` and `per_sample_transform_on_device`"):
        val_preprocessor = data_pipeline.worker_preprocessor(RunningStage.VALIDATING)
    with pytest.raises(MisconfigurationException, match="`per_batch_transform` and `per_sample_transform_on_device`"):
        test_preprocessor = data_pipeline.worker_preprocessor(RunningStage.TESTING)
    predict_preprocessor = data_pipeline.worker_preprocessor(RunningStage.PREDICTING)

    assert train_preprocessor.collate_fn.func == preprocess.collate
    assert predict_preprocessor.collate_fn.func == DataPipeline._identity


def test_iterable_auto_dataset(tmpdir):
    class CustomInput(BaseInput):
        def load_sample(self, index: int) -> Dict[str, int]:
            return {"index": index}

    ds = IterableAutoDataset(range(10), data_source=CustomInput(), running_stage=RunningStage.TRAINING)

    for index, v in enumerate(ds):
        assert v == {"index": index}


class CustomPreprocessHyperparameters(DefaultPreprocess):
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
    preprocess = CustomPreprocessHyperparameters("token", **kwargs)
    state_dict = preprocess.state_dict()
    torch.save(state_dict, os.path.join(tmpdir, "state_dict.pt"))
    state_dict = torch.load(os.path.join(tmpdir, "state_dict.pt"))
    preprocess = CustomPreprocessHyperparameters.load_from_state_dict(state_dict)
    assert isinstance(preprocess, CustomPreprocessHyperparameters)

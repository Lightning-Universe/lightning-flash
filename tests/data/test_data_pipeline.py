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

from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from flash.core import Task
from flash.data.auto_dataset import AutoDataset
from flash.data.batch import _PostProcessor, _PreProcessor
from flash.data.data_module import DataModule
from flash.data.data_pipeline import _StageOrchestrator, DataPipeline
from flash.data.process import Postprocess, Preprocess


class DummyDataset(torch.utils.data.Dataset):

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.rand(1), torch.rand(1)

    def __len__(self) -> int:
        return 5


class CustomModel(Task):

    def __init__(self, postprocess: Optional[Postprocess] = None):
        super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())
        self._postprocess = postprocess

    def train_dataloader(self) -> Any:
        return DataLoader(DummyDataset())


class CustomDataModule(DataModule):

    def __init__(self):
        super().__init__(
            train_ds=DummyDataset(),
            valid_ds=DummyDataset(),
            test_ds=DummyDataset(),
            predict_ds=DummyDataset(),
        )


@pytest.mark.skipif(reason="Still using DataPipeline Old API")
@pytest.mark.parametrize("use_preprocess", [False, True])
@pytest.mark.parametrize("use_postprocess", [False, True])
def test_data_pipeline_init_and_assignement(use_preprocess, use_postprocess, tmpdir):

    class SubPreprocess(Preprocess):
        pass

    class SubPostprocess(Postprocess):
        pass

    data_pipeline = DataPipeline(
        SubPreprocess() if use_preprocess else None,
        SubPostprocess() if use_postprocess else None,
    )
    assert isinstance(data_pipeline._preprocess_pipeline, SubPreprocess if use_preprocess else Preprocess)
    assert isinstance(data_pipeline._postprocess_pipeline, SubPostprocess if use_postprocess else Postprocess)

    model = CustomModel(Postprocess())
    model.data_pipeline = data_pipeline
    assert isinstance(model._preprocess, Preprocess)
    assert isinstance(model._postprocess, SubPostprocess if use_postprocess else Postprocess)


def test_data_pipeline_is_overriden_and_resolve_function_hierarchy(tmpdir):

    class CustomPreprocess(Preprocess):

        def load_data(self, *_, **__):
            return 0

        def test_load_data(self, *_, **__):
            return 1

        def predict_load_data(self, *_, **__):
            return 2

        def predict_load_sample(self, *_, **__):
            return 3

        def val_load_sample(self, *_, **__):
            return 4

        def val_per_sample_pre_tensor_transform(self, *_, **__):
            return 5

        def predict_per_sample_to_tensor_transform(self, *_, **__):
            return 7

        def train_per_sample_post_tensor_transform(self, *_, **__):
            return 8

        def test_collate(self, *_, **__):
            return 9

        def val_per_sample_transform_on_device(self, *_, **__):
            return 10

        def train_per_batch_transform_on_device(self, *_, **__):
            return 11

        def test_per_batch_transform_on_device(self, *_, **__):
            return 12

    preprocess = CustomPreprocess()
    data_pipeline = DataPipeline(preprocess)
    train_func_names = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._preprocess_pipeline, RunningStage.TRAINING, Preprocess
        )
        for k in data_pipeline.PREPROCESS_FUNCS
    }
    val_func_names = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._preprocess_pipeline, RunningStage.VALIDATING, Preprocess
        )
        for k in data_pipeline.PREPROCESS_FUNCS
    }
    test_func_names = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._preprocess_pipeline, RunningStage.TESTING, Preprocess
        )
        for k in data_pipeline.PREPROCESS_FUNCS
    }
    predict_func_names = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._preprocess_pipeline, RunningStage.PREDICTING, Preprocess
        )
        for k in data_pipeline.PREPROCESS_FUNCS
    }
    # load_data
    assert train_func_names["load_data"] == "load_data"
    assert val_func_names["load_data"] == "load_data"
    assert test_func_names["load_data"] == "test_load_data"
    assert predict_func_names["load_data"] == "predict_load_data"

    # load_sample
    assert train_func_names["load_sample"] == "load_sample"
    assert val_func_names["load_sample"] == "val_load_sample"
    assert test_func_names["load_sample"] == "load_sample"
    assert predict_func_names["load_sample"] == "predict_load_sample"

    # per_sample_pre_tensor_transform
    assert train_func_names["per_sample_pre_tensor_transform"] == "per_sample_pre_tensor_transform"
    assert val_func_names["per_sample_pre_tensor_transform"] == "val_per_sample_pre_tensor_transform"
    assert test_func_names["per_sample_pre_tensor_transform"] == "per_sample_pre_tensor_transform"
    assert predict_func_names["per_sample_pre_tensor_transform"] == "per_sample_pre_tensor_transform"

    # per_sample_to_tensor_transform
    assert train_func_names["per_sample_to_tensor_transform"] == "per_sample_to_tensor_transform"
    assert val_func_names["per_sample_to_tensor_transform"] == "per_sample_to_tensor_transform"
    assert test_func_names["per_sample_to_tensor_transform"] == "per_sample_to_tensor_transform"
    assert predict_func_names["per_sample_to_tensor_transform"] == "predict_per_sample_to_tensor_transform"

    # per_sample_post_tensor_transform
    assert train_func_names["per_sample_post_tensor_transform"] == "train_per_sample_post_tensor_transform"
    assert val_func_names["per_sample_post_tensor_transform"] == "per_sample_post_tensor_transform"
    assert test_func_names["per_sample_post_tensor_transform"] == "per_sample_post_tensor_transform"
    assert predict_func_names["per_sample_post_tensor_transform"] == "per_sample_post_tensor_transform"

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
    assert _seq.per_sample_pre_tensor_transform.func == preprocess.per_sample_pre_tensor_transform
    assert _seq.per_sample_to_tensor_transform.func == preprocess.per_sample_to_tensor_transform
    assert _seq.per_sample_post_tensor_transform.func == preprocess.train_per_sample_post_tensor_transform
    assert train_worker_preprocessor.collate_fn.func == default_collate
    assert train_worker_preprocessor.per_batch_transform.func == preprocess.per_batch_transform

    _seq = val_worker_preprocessor.per_sample_transform
    assert _seq.per_sample_pre_tensor_transform.func == preprocess.val_per_sample_pre_tensor_transform
    assert _seq.per_sample_to_tensor_transform.func == preprocess.per_sample_to_tensor_transform
    assert _seq.per_sample_post_tensor_transform.func == preprocess.per_sample_post_tensor_transform
    assert val_worker_preprocessor.collate_fn.func == data_pipeline._do_nothing_collate
    assert val_worker_preprocessor.per_batch_transform.func == preprocess.per_batch_transform

    _seq = test_worker_preprocessor.per_sample_transform
    assert _seq.per_sample_pre_tensor_transform.func == preprocess.per_sample_pre_tensor_transform
    assert _seq.per_sample_to_tensor_transform.func == preprocess.per_sample_to_tensor_transform
    assert _seq.per_sample_post_tensor_transform.func == preprocess.per_sample_post_tensor_transform
    assert test_worker_preprocessor.collate_fn.func == preprocess.test_collate
    assert test_worker_preprocessor.per_batch_transform.func == preprocess.per_batch_transform

    _seq = predict_worker_preprocessor.per_sample_transform
    assert _seq.per_sample_pre_tensor_transform.func == preprocess.per_sample_pre_tensor_transform
    assert _seq.per_sample_to_tensor_transform.func == preprocess.predict_per_sample_to_tensor_transform
    assert _seq.per_sample_post_tensor_transform.func == preprocess.per_sample_post_tensor_transform
    assert predict_worker_preprocessor.collate_fn.func == default_collate
    assert predict_worker_preprocessor.per_batch_transform.func == preprocess.per_batch_transform


class CustomPreprocess(Preprocess):

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
    data_pipeline = DataPipeline(preprocess)

    data_pipeline.worker_preprocessor(RunningStage.TRAINING)
    with pytest.raises(MisconfigurationException, match="are mutual exclusive"):
        data_pipeline.worker_preprocessor(RunningStage.VALIDATING)
    with pytest.raises(MisconfigurationException, match="are mutual exclusive"):
        data_pipeline.worker_preprocessor(RunningStage.TESTING)
    data_pipeline.worker_preprocessor(RunningStage.PREDICTING)


@pytest.mark.skipif(reason="Still using DataPipeline Old API")
def test_detach_preprocessing_from_model(tmpdir):

    preprocess = CustomPreprocess()
    data_pipeline = DataPipeline(preprocess)
    model = CustomModel()
    model.data_pipeline = data_pipeline

    assert model.train_dataloader().collate_fn == default_collate
    assert model.transfer_batch_to_device.__self__ == model
    model.on_train_dataloader()
    assert isinstance(model.train_dataloader().collate_fn, _PreProcessor)
    assert isinstance(model.transfer_batch_to_device, _StageOrchestrator)
    model.on_fit_end()
    assert model.transfer_batch_to_device.__self__ == model
    assert model.train_dataloader().collate_fn == default_collate


class TestPreprocess(Preprocess):

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


@pytest.mark.skipif(reason="Still using DataPipeline Old API")
def test_attaching_datapipeline_to_model(tmpdir):

    preprocess = TestPreprocess()
    data_pipeline = DataPipeline(preprocess)

    class TestModel(CustomModel):

        stages = [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]
        on_train_start_called = False
        on_val_start_called = False
        on_test_start_called = False
        on_predict_start_called = False

        def on_fit_start(self):
            assert self.predict_step.__self__ == self
            self._saved_predict_step = self.predict_step

        def _compare_pre_processor(self, p1, p2):
            p1_seq = p1.per_sample_transform
            p2_seq = p2.per_sample_transform
            assert p1_seq.per_sample_pre_tensor_transform.func == p2_seq.per_sample_pre_tensor_transform.func
            assert p1_seq.per_sample_to_tensor_transform.func == p2_seq.per_sample_to_tensor_transform.func
            assert p1_seq.per_sample_post_tensor_transform.func == p2_seq.per_sample_post_tensor_transform.func
            assert p1.collate_fn.func == p2.collate_fn.func
            assert p1.per_batch_transform.func == p2.per_batch_transform.func

        def _assert_stage_orchestrator_state(
            self, stage_mapping: Dict, current_running_stage: RunningStage, cls=_PreProcessor
        ):
            assert isinstance(stage_mapping[current_running_stage], cls)
            assert stage_mapping[current_running_stage] is not None

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
                self.predict_step._stage_mapping, current_running_stage, cls=_PostProcessor
            )

        def on_fit_end(self) -> None:
            super().on_fit_end()
            assert self.train_dataloader().collate_fn == default_collate
            assert self.val_dataloader().collate_fn == default_collate
            assert self.test_dataloader().collate_fn == default_collate
            assert self.predict_dataloader().collate_fn == default_collate
            assert not isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            assert self.predict_step == self._saved_predict_step

    datamodule = CustomDataModule()
    datamodule._data_pipeline = data_pipeline
    model = TestModel()
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=datamodule)
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

        def _attach_postprocess_to_model(self, model: 'Task', _postprocesssor: _PostProcessor) -> 'Task':
            model.predict_step = self._model_predict_step_wrapper(model.predict_step, _postprocesssor, model)
            return model

    data_pipeline = CustomDataPipeline(preprocess)
    _postprocesssor = data_pipeline._create_uncollate_postprocessors()
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


class TestPreprocessTransformations(Preprocess):

    def __init__(self):
        super().__init__()

        self.train_load_data_called = False
        self.train_per_sample_pre_tensor_transform_called = False
        self.train_collate_called = False
        self.train_per_batch_transform_on_device_called = False
        self.val_load_data_called = False
        self.val_load_sample_called = False
        self.val_per_sample_to_tensor_transform_called = False
        self.val_collate_called = False
        self.val_per_batch_transform_on_device_called = False
        self.test_load_data_called = False
        self.test_per_sample_to_tensor_transform_called = False
        self.test_per_sample_post_tensor_transform_called = False
        self.predict_load_data_called = False

    def train_load_data(self, sample) -> LamdaDummyDataset:
        self.train_load_data_called = True
        return LamdaDummyDataset(lambda: (0, 1, 2, 3))

    def train_per_sample_pre_tensor_transform(self, sample: Any) -> Any:
        self.train_per_sample_pre_tensor_transform_called = True
        return sample + (5, )

    def train_collate(self, samples) -> torch.Tensor:
        self.train_collate_called = True
        return torch.tensor([list(s) for s in samples])

    def train_per_batch_transform_on_device(self, batch: Any) -> Any:
        self.train_per_batch_transform_on_device_called = True
        assert torch.equal(batch, torch.tensor([[0, 1, 2, 3, 5], [0, 1, 2, 3, 5]]))

    def val_load_data(self, sample, dataset) -> List[int]:
        self.val_load_data_called = True
        assert isinstance(dataset, AutoDataset)
        return list(range(5))

    def val_load_sample(self, sample) -> Dict[str, torch.Tensor]:
        self.val_load_sample_called = True
        return {"a": sample, "b": sample + 1}

    def val_per_sample_to_tensor_transform(self, sample: Any) -> torch.Tensor:
        self.val_per_sample_to_tensor_transform_called = True
        return sample

    def val_collate(self, samples) -> Dict[str, torch.Tensor]:
        self.val_collate_called = True
        _count = samples[0]['a']
        assert samples == [{'a': _count, 'b': _count + 1}, {'a': _count + 1, 'b': _count + 2}]
        return {'a': torch.tensor([0, 1]), 'b': torch.tensor([1, 2])}

    def val_per_batch_transform_on_device(self, batch: Any) -> Any:
        self.val_per_batch_transform_on_device_called = True
        batch = batch[0]
        assert torch.equal(batch["a"], torch.tensor([0, 1]))
        assert torch.equal(batch["b"], torch.tensor([1, 2]))
        return [False]

    def test_load_data(self, sample) -> LamdaDummyDataset:
        self.test_load_data_called = True
        return LamdaDummyDataset(lambda: [torch.rand(1), torch.rand(1)])

    def test_per_sample_to_tensor_transform(self, sample: Any) -> torch.Tensor:
        self.test_per_sample_to_tensor_transform_called = True
        return sample

    def test_per_sample_post_tensor_transform(self, sample: torch.Tensor) -> torch.Tensor:
        self.test_per_sample_post_tensor_transform_called = True
        return sample

    def predict_load_data(self, sample) -> LamdaDummyDataset:
        self.predict_load_data_called = True
        return LamdaDummyDataset(lambda: (["a", "b"]))


class TestPreprocessTransformations2(TestPreprocessTransformations):

    def val_per_sample_to_tensor_transform(self, sample: Any) -> torch.Tensor:
        self.val_per_sample_to_tensor_transform_called = True
        return {"a": torch.tensor(sample["a"]), "b": torch.tensor(sample["b"])}


@pytest.mark.skipif(reason="Still using DataPipeline Old API")
def test_datapipeline_transformations(tmpdir):

    class CustomModel(Task):

        def __init__(self):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())

        def training_step(self, batch, batch_idx):
            assert batch is None

        def validation_step(self, batch, batch_idx):
            assert batch is False

        def test_step(self, batch, batch_idx):
            assert len(batch) == 2
            assert batch[0].shape == torch.Size([2, 1])

        def predict_step(self, batch, batch_idx, dataloader_idx):
            assert batch == [('a', 'a'), ('b', 'b')]
            return torch.tensor([0, 0, 0])

    class CustomDataModule(DataModule):

        preprocess_cls = TestPreprocessTransformations

    datamodule = CustomDataModule.from_load_data_inputs(1, 1, 1, 1, batch_size=2)

    assert datamodule.train_dataloader().dataset[0] == (0, 1, 2, 3)
    batch = next(iter(datamodule.train_dataloader()))
    assert torch.equal(batch, torch.tensor([[0, 1, 2, 3, 5], [0, 1, 2, 3, 5]]))

    assert datamodule.val_dataloader().dataset[0] == {'a': 0, 'b': 1}
    assert datamodule.val_dataloader().dataset[1] == {'a': 1, 'b': 2}
    with pytest.raises(MisconfigurationException, match="When ``per_sample_to_tensor_transform``"):
        batch = next(iter(datamodule.val_dataloader()))

    CustomDataModule.preprocess_cls = TestPreprocessTransformations2
    datamodule = CustomDataModule.from_load_data_inputs(1, 1, 1, 1, batch_size=2)
    batch = next(iter(datamodule.val_dataloader()))
    assert torch.equal(batch["a"], torch.tensor([0, 1]))
    assert torch.equal(batch["b"], torch.tensor([1, 2]))

    model = CustomModel()
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        limit_test_batches=2,
        limit_predict_batches=2,
        num_sanity_val_steps=1
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model)
    trainer.predict(model)

    # todo (tchaton) resolve the lost reference.
    preprocess = model._preprocess
    # assert preprocess.train_load_data_called
    # assert preprocess.train_per_sample_pre_tensor_transform_called
    # assert preprocess.train_collate_called
    assert preprocess.train_per_batch_transform_on_device_called
    # assert preprocess.val_load_data_called
    # assert preprocess.val_load_sample_called
    # assert preprocess.val_per_sample_to_tensor_transform_called
    # assert preprocess.val_collate_called
    assert preprocess.val_per_batch_transform_on_device_called
    # assert preprocess.test_load_data_called
    # assert preprocess.test_per_sample_to_tensor_transform_called
    # assert preprocess.test_per_sample_post_tensor_transform_called
    # assert preprocess.predict_load_data_called


def test_is_overriden_recursive(tmpdir):

    class TestPreprocess(Preprocess):

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


@pytest.mark.skipif(reason="Still using DataPipeline Old API")
@mock.patch("torch.save")  # need to mock torch.save or we get pickle error
def test_dummy_example(tmpdir):

    class ImageClassificationPreprocess(Preprocess):

        def __init__(self, to_tensor_transform, train_per_sample_transform_on_device):
            super().__init__()
            self._to_tensor = to_tensor_transform
            self._train_per_sample_transform_on_device = train_per_sample_transform_on_device

        def load_data(self, folder: str):
            # from folder -> return files paths
            return ["a.jpg", "b.jpg"]

        def load_sample(self, path: str) -> Image.Image:
            # from a file path, load the associated image
            img8Bit = np.uint8(np.random.uniform(0, 1, (64, 64, 3)) * 255.0)
            return Image.fromarray(img8Bit)

        def per_sample_to_tensor_transform(self, pil_image: Image.Image) -> torch.Tensor:
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

        @property
        def preprocess(self):
            return self.preprocess_cls(self.to_tensor_transform, self.train_per_sample_transform_on_device)

        @classmethod
        def from_folders(
            cls, train_folder: Optional[str], val_folder: Optional[str], test_folder: Optional[str],
            predict_folder: Optional[str], to_tensor_transform: torch.nn.Module,
            train_per_sample_transform_on_device: torch.nn.Module, batch_size: int
        ):

            # attach the arguments for the preprocess onto the cls
            cls.to_tensor_transform = to_tensor_transform
            cls.train_per_sample_transform_on_device = train_per_sample_transform_on_device

            # call ``from_load_data_inputs``
            return cls.from_load_data_inputs(
                train_load_data_input=train_folder,
                valid_load_data_input=val_folder,
                test_load_data_input=test_folder,
                predict_load_data_input=predict_folder,
                batch_size=batch_size
            )

    datamodule = CustomDataModule.from_folders(
        "train_folder", "val_folder", "test_folder", None, T.ToTensor(), T.RandomHorizontalFlip(), batch_size=2
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
        num_sanity_val_steps=1
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model)

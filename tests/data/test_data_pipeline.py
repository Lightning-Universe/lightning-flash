from typing import Any, Callable, Dict, Optional

import pytest
import torch
from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from flash.core import Task
from flash.data.batch import _PostProcessor, _PreProcessor
from flash.data.data_module import DataModule
from flash.data.data_pipeline import _StageOrchestrator, DataPipeline
from flash.data.process import Postprocess, Preprocess
from tests.vision.detection.test_model import collate_fn


class DummyDataset(torch.utils.data.Dataset):

    def __getitem__(self, index: int) -> Any:
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

        def validation_load_sample(self, *_, **__):
            return 4

        def predict_per_sample_transform(self, *_, **__):
            return 5

        def test_collate(self, *_, **__):
            return 6

        def validation_per_sample_transform_on_device(self, *_, **__):
            return 7

        def train_per_batch_transform_on_device(self, *_, **__):
            return 8

        def test_per_batch_transform_on_device(self, *_, **__):
            return 8

    preprocess = CustomPreprocess()
    data_pipeline = DataPipeline(preprocess)
    train_func_names = {
        k: data_pipeline._resolve_function_hierarchy(
            k, data_pipeline._preprocess_pipeline, RunningStage.TRAINING, Preprocess
        )
        for k in data_pipeline.PREPROCESS_FUNCS
    }
    validation_func_names = {
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
    assert validation_func_names["load_data"] == "load_data"
    assert test_func_names["load_data"] == "test_load_data"
    assert predict_func_names["load_data"] == "predict_load_data"

    # load_sample
    assert train_func_names["load_sample"] == "load_sample"
    assert validation_func_names["load_sample"] == "validation_load_sample"
    assert test_func_names["load_sample"] == "load_sample"
    assert predict_func_names["load_sample"] == "predict_load_sample"

    # per_sample_transform
    assert train_func_names["per_sample_transform"] == "per_sample_transform"
    assert validation_func_names["per_sample_transform"] == "per_sample_transform"
    assert test_func_names["per_sample_transform"] == "per_sample_transform"
    assert predict_func_names["per_sample_transform"] == "predict_per_sample_transform"

    # collate
    assert train_func_names["collate"] == "collate"
    assert validation_func_names["collate"] == "collate"
    assert test_func_names["collate"] == "test_collate"
    assert predict_func_names["collate"] == "collate"

    # per_sample_transform_on_device
    assert train_func_names["per_sample_transform_on_device"] == "per_sample_transform_on_device"
    assert validation_func_names["per_sample_transform_on_device"] == "validation_per_sample_transform_on_device"
    assert test_func_names["per_sample_transform_on_device"] == "per_sample_transform_on_device"
    assert predict_func_names["per_sample_transform_on_device"] == "per_sample_transform_on_device"

    # per_batch_transform_on_device
    assert train_func_names["per_batch_transform_on_device"] == "train_per_batch_transform_on_device"
    assert validation_func_names["per_batch_transform_on_device"] == "per_batch_transform_on_device"
    assert test_func_names["per_batch_transform_on_device"] == "test_per_batch_transform_on_device"
    assert predict_func_names["per_batch_transform_on_device"] == "per_batch_transform_on_device"

    train_worker_preprocessor = data_pipeline.worker_preprocessor(RunningStage.TRAINING)
    validation_worker_preprocessor = data_pipeline.worker_preprocessor(RunningStage.VALIDATING)
    test_worker_preprocessor = data_pipeline.worker_preprocessor(RunningStage.TESTING)
    predict_worker_preprocessor = data_pipeline.worker_preprocessor(RunningStage.PREDICTING)

    assert train_worker_preprocessor.per_sample_transform.func == preprocess.per_sample_transform
    assert train_worker_preprocessor.collate_fn.func == default_collate
    assert train_worker_preprocessor.per_batch_transform.func == preprocess.per_batch_transform

    assert validation_worker_preprocessor.per_sample_transform.func == preprocess.per_sample_transform
    assert validation_worker_preprocessor.collate_fn.func == data_pipeline._do_nothing_collate
    assert validation_worker_preprocessor.per_batch_transform.func == preprocess.per_batch_transform

    assert test_worker_preprocessor.per_sample_transform.func == preprocess.per_sample_transform
    assert test_worker_preprocessor.collate_fn.func == preprocess.test_collate
    assert test_worker_preprocessor.per_batch_transform.func == preprocess.per_batch_transform

    assert predict_worker_preprocessor.per_sample_transform.func == preprocess.predict_per_sample_transform
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

    def validation_per_batch_transform(self, *_, **__):
        pass

    def validation_per_sample_transform_on_device(self, *_, **__):
        pass

    def predict_per_sample_transform(self, *_, **__):
        pass

    def predict_per_sample_transform_on_device(self, *_, **__):
        pass

    def predict_per_batch_transform_on_device(self, *_, **__):
        pass


def test_data_pipeline_predict_worker_preprocessor_and_device_preprocessor(tmpdir):

    preprocess = CustomPreprocess()
    data_pipeline = DataPipeline(preprocess)

    _ = data_pipeline.worker_preprocessor(RunningStage.TRAINING)
    with pytest.raises(MisconfigurationException, match="are mutual exclusive"):
        _ = data_pipeline.worker_preprocessor(RunningStage.VALIDATING)
    with pytest.raises(MisconfigurationException, match="are mutual exclusive"):
        _ = data_pipeline.worker_preprocessor(RunningStage.TESTING)
    _ = data_pipeline.worker_preprocessor(RunningStage.PREDICTING)


def test_detach_preprocessing_from_model(tmpdir):

    preprocess = CustomPreprocess()
    data_pipeline = DataPipeline(preprocess)
    model = CustomModel()
    model.data_pipeline = data_pipeline

    assert model.train_dataloader().collate_fn == default_collate
    assert model.transfer_batch_to_device.__self__ == model
    model.on_train_start()
    assert isinstance(model.train_dataloader().collate_fn, _PreProcessor)
    assert isinstance(model.transfer_batch_to_device, _StageOrchestrator)
    model.on_train_end()
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

    def validation_per_sample_transform_on_device(self, *_, **__):
        pass

    def predict_per_sample_transform(self, *_, **__):
        pass

    def predict_per_sample_transform_on_device(self, *_, **__):
        pass

    def predict_per_batch_transform_on_device(self, *_, **__):
        pass


def test_attaching_datapipeline_to_model(tmpdir):

    preprocess = TestPreprocess()
    data_pipeline = DataPipeline(preprocess)

    class TestModel(CustomModel):

        stages = [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]
        on_train_start_called = False
        on_validation_start_called = False
        on_test_start_called = False
        on_predict_start_called = False

        def on_fit_start(self):
            assert self.predict_step.__self__ == self
            self._saved_predict_step = self.predict_step

        def _compare_pre_processor(self, p1, p2):
            assert p1.per_sample_transform.func == p2.per_sample_transform.func
            assert p1.collate_fn.func == p2.collate_fn.func
            assert p1.per_batch_transform.func == p2.per_batch_transform.func

        def _assert_stage_orchestrator_state(
            self, stage_mapping: Dict, current_running_stage: RunningStage, cls=_PreProcessor
        ):
            assert isinstance(stage_mapping[current_running_stage], cls)
            for stage in [s for s in self.stages if s != current_running_stage]:
                assert stage_mapping[stage] is None

        def on_train_start(self) -> None:
            current_running_stage = RunningStage.TRAINING
            self.on_train_start_called = True
            collate_fn = self.train_dataloader().collate_fn
            assert collate_fn == default_collate
            assert not isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            super().on_train_start()
            collate_fn = self.train_dataloader().collate_fn  # noqa F811
            assert collate_fn._stage == current_running_stage
            self._compare_pre_processor(collate_fn, self.data_pipeline.worker_preprocessor(current_running_stage))
            assert isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            self._assert_stage_orchestrator_state(self.transfer_batch_to_device._stage_mapping, current_running_stage)

        def on_validation_start(self) -> None:
            current_running_stage = RunningStage.VALIDATING
            self.on_validation_start_called = True
            collate_fn = self.val_dataloader().collate_fn
            assert collate_fn == default_collate
            assert not isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            super().on_validation_start()
            collate_fn = self.val_dataloader().collate_fn  # noqa F811
            assert collate_fn._stage == current_running_stage
            self._compare_pre_processor(collate_fn, self.data_pipeline.worker_preprocessor(current_running_stage))
            assert isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            self._assert_stage_orchestrator_state(self.transfer_batch_to_device._stage_mapping, current_running_stage)

        def on_test_start(self) -> None:
            current_running_stage = RunningStage.TESTING
            self.on_test_start_called = True
            collate_fn = self.test_dataloader().collate_fn
            assert collate_fn == default_collate
            assert not isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            super().on_test_start()
            collate_fn = self.test_dataloader().collate_fn  # noqa F811
            assert collate_fn._stage == current_running_stage
            self._compare_pre_processor(collate_fn, self.data_pipeline.worker_preprocessor(current_running_stage))
            assert isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            self._assert_stage_orchestrator_state(self.transfer_batch_to_device._stage_mapping, current_running_stage)

        def on_predict_start(self) -> None:
            current_running_stage = RunningStage.PREDICTING
            self.on_predict_start_called = True
            collate_fn = self.predict_dataloader().collate_fn
            assert collate_fn == default_collate
            assert not isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            assert self.predict_step == self._saved_predict_step
            super().on_predict_start()
            collate_fn = self.predict_dataloader().collate_fn  # noqa F811
            assert collate_fn._stage == current_running_stage
            self._compare_pre_processor(collate_fn, self.data_pipeline.worker_preprocessor(current_running_stage))
            assert isinstance(self.transfer_batch_to_device, _StageOrchestrator)
            assert isinstance(self.predict_step, _StageOrchestrator)
            self._assert_stage_orchestrator_state(self.transfer_batch_to_device._stage_mapping, current_running_stage)
            self._assert_stage_orchestrator_state(
                self.predict_step._stage_mapping, current_running_stage, cls=_PostProcessor
            )

        def on_fit_end(self) -> None:
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

    assert model.on_train_start_called
    assert model.on_validation_start_called
    assert model.on_test_start_called
    assert model.on_predict_start_called


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

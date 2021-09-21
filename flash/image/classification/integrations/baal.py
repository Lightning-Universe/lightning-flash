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
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerStatus
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

import flash
from flash import DataModule
from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.data.data_pipeline import DataPipeline
from flash.core.utilities.imports import _BAAL_AVAILABLE, requires

if _BAAL_AVAILABLE:
    from baal.active.dataset import ActiveLearningDataset
    from baal.active.heuristics import AbstractHeuristic, BALD


def dataset_to_non_labelled_tensor(dataset: BaseAutoDataset) -> torch.tensor:
    return torch.zeros(len(dataset))


class ActiveLearningDataModule(LightningDataModule):
    @requires("baal")
    def __init__(
        self,
        labelled: Optional[DataModule] = None,
        unlabelled: Optional[DataModule] = None,
        heuristic: "AbstractHeuristic" = BALD(reduction=np.mean),
        map_dataset_to_labelled: Optional[Callable] = dataset_to_non_labelled_tensor,
    ):
        self.labelled = labelled
        self.unlabelled = unlabelled
        self.heuristic = heuristic
        self.map_dataset_to_labelled = map_dataset_to_labelled

        if self.unlabelled:
            raise MisconfigurationException("The unlabelled `datamodule` isn't support yet.")

        if self.labelled and (
            self.labelled._val_ds is not None
            or self.labelled._test_ds is not None
            or self.labelled._predict_ds is not None
        ):
            raise MisconfigurationException("The labelled `datamodule` should have only train data.")

        self._dataset: Optional[ActiveLearningDataset] = None

        self._initialize_active_learning_dataset()

    def _initialize_active_learning_dataset(self):
        if self.labelled and self.unlabelled:
            raise MisconfigurationException("This isn't supported yet")

        if self.labelled:
            if not self.labelled.num_classes:
                raise MisconfigurationException("The labelled dataset should be labelled")

            dataset = self.labelled._train_ds
            self._dataset = ActiveLearningDataset(dataset, labelled=self.map_dataset_to_labelled(dataset))

            if not len(self._dataset):
                self.label(indices=[0])

    @property
    def num_classes(self) -> Optional[int]:
        return getattr(self.labelled, "num_classes", None) or getattr(self.unlabelled, "num_classes", None)

    @property
    def data_pipeline(self) -> "DataPipeline":
        if self.labelled:
            return self.labelled.data_pipeline
        return self.unlabelled.data_pipeline

    def train_dataloader(self) -> "DataLoader":
        if self.labelled:
            self.labelled._train_ds = self._dataset
            return self.labelled.train_dataloader()
        raise NotImplementedError

    def predict_dataloader(self) -> "DataLoader":
        if self.labelled:
            self.labelled._train_ds = self._dataset.pool
            return self.labelled.train_dataloader()
        raise NotImplementedError

    def label(self, predictions: Any = None, indices=None):
        if predictions and indices:
            raise MisconfigurationException(
                "The `predictions` and `indices` are mutually exclusive, pass only of one them."
            )
        if predictions:
            uncertainties = [self.heuristic.get_uncertainties(np.asarray(p)) for idx, p in enumerate(predictions)]
            indices = np.argsort(uncertainties)
        if self._dataset is not None:
            self._dataset.labelled[indices] = True

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self._dataset.state_dict()

    def load_state_dict(self, state_dict) -> None:
        return self._dataset.load_state_dict(state_dict)


class ActiveLearningTrainer(flash.Trainer):
    def __init__(self, *args, **kwags):
        kwags["reload_dataloaders_every_n_epochs"] = 1
        super().__init__(*args, **kwags)

        active_learning_loop = ActiveLearningLoop(label_epoch_frequency=1)
        active_learning_loop.connect(self.fit_loop)
        self.fit_loop = active_learning_loop
        active_learning_loop.trainer = self


class ActiveLearningLoop(Loop):
    @requires("baal")
    def __init__(self, label_epoch_frequency: int, inference_iteration: int = 2):
        super().__init__()
        self.label_epoch_frequency = label_epoch_frequency
        self.inference_iteration = inference_iteration

        self.fit_loop: Optional[FitLoop] = None
        self.progress = Progress()

        self._should_continue: bool = False
        self._model_state_dict: Optional[Dict[str, torch.Tensor]] = None

    @property
    def done(self) -> bool:
        return self.progress.current.completed > self.max_epochs or self._should_continue

    def connect(self, fit_loop: FitLoop):
        self.fit_loop = fit_loop
        self.max_epochs = self.fit_loop.max_epochs
        self.fit_loop.max_epochs = self.label_epoch_frequency

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        self.trainer.predict_loop._return_predictions = True
        self._model_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def reset(self) -> None:
        if self.restarting:
            self.progress.current.reset_on_restart()

            self.datamodule.load_state_dict()

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        # This is a hack and we need to clean this on the Lightning side.
        self._reset_dataloader_for_stage(RunningStage.TRAINING)
        self._reset_dataloader_for_stage(RunningStage.PREDICTING)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self.progress.increment_started()
        self.fit_loop.run()
        self._reset_predicting()
        predictions = self.trainer.predict_loop.run()
        self.trainer.datamodule.label(predictions=predictions)
        self._reset_fitting()
        self.progress.increment_processed()

    def on_advance_end(self) -> None:
        self.trainer.lightning_module.load_state_dict(self._model_state_dict)
        self.progress.increment_completed()
        return super().on_advance_end()

    def on_run_end(self):
        self.trainer.lightning_module.predict_step = self.trainer.lightning_module._predict_step
        return super().on_run_end()

    def on_save_checkpoint(self) -> Dict:
        return {"datamodule_state_dict": self.trainer.datamodule.state_dict()}

    def on_load_checkpoint(self, state_dict) -> None:
        self.trainer.datamodule.load_state_dict(state_dict.pop("datamodule_state_dict"))

    def __getattr__(self, key):
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def _reset_fitting(self):
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.state.status = TrainerStatus.RUNNING
        self.trainer.training = True
        self.trainer.lightning_module.on_train_dataloader()

    def _reset_predicting(self):
        self.trainer.state.fn = TrainerFn.PREDICTING
        self.trainer.state.status = TrainerStatus.RUNNING
        self.trainer.predicting = True
        self.trainer.lightning_module.on_predict_dataloader()
        self._enable_mc_dropout()

    def _reset_dataloader_for_stage(self, running_state: RunningStage):
        dataloader_name = f"{running_state.value}_dataloader"
        setattr(
            self.trainer.lightning_module,
            dataloader_name,
            _PatchDataLoader(getattr(self.trainer.datamodule, dataloader_name)(), running_state),
        )
        setattr(self.trainer, dataloader_name, None)
        getattr(self.trainer, f"reset_{dataloader_name}")(self.trainer.lightning_module)

    def _enable_mc_dropout(self):
        # prevent the model to put into val model - hack
        self.trainer.lightning_module.on_predict_model_eval = self._do_nothing
        for _, module in self.trainer.lightning_module.named_modules():
            if isinstance(module, _BatchNorm):
                module.eval()
        # save the predict_step and replace it within a mc_inference function
        if getattr(self.trainer.lightning_module, "_predict_step", None) is None:
            self.trainer.lightning_module._predict_step = self.trainer.lightning_module.predict_step
        self.trainer.lightning_module.predict_step = partial(
            self._mc_inference,
            predict_step_fn=self.trainer.lightning_module._predict_step,
            inference_iteration=self.inference_iteration,
        )

    def _do_nothing(self, *args, **kwargs):
        pass

    @staticmethod
    def _mc_inference(*args, predict_step_fn: Callable = None, inference_iteration: int = None, **kwargs):
        out = []
        for _ in range(inference_iteration):
            out.append(predict_step_fn(*args, **kwargs))
        return out

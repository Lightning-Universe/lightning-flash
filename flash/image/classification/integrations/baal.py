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
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerStatus
from torch.nn.modules.batchnorm import _BatchNorm

import flash
from flash.core.utilities.imports import _BAAL_AVAILABLE, requires

if _BAAL_AVAILABLE:
    from baal.active import ActiveLearningDataset
    from baal.active.heuristics import AbstractHeuristic, BALD


class ActiveLearningTrainer(flash.Trainer):
    def __init__(self, *args, **kwags):
        super().__init__(*args, **kwags)

        active_learning_loop = ActiveLearningLoop(label_epoch_frequency=1, heuristics=BALD(reduction=np.mean))
        active_learning_loop.connect(self.fit_loop)
        self.fit_loop = active_learning_loop
        active_learning_loop.trainer = self


class ActiveLearningLoop(Loop):
    @requires("baal")
    def __init__(self, label_epoch_frequency: int, heuristics: "AbstractHeuristic", inference_iteration: int = 2):
        super().__init__()
        self.label_epoch_frequency = label_epoch_frequency
        self.fit_loop: Optional[FitLoop] = None
        self.state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.progress = Progress()
        self.heuristics = heuristics
        self._should_continue: bool = False
        self.inference_iteration = inference_iteration

    def __getattr__(self, key):
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def connect(self, fit_loop: FitLoop):
        self.fit_loop = fit_loop
        self.max_epochs = self.fit_loop.max_epochs
        self.fit_loop.max_epochs = self.label_epoch_frequency

    def _to_active_dataset(self, dataset):
        return ActiveLearningDataset(dataset, labelled=torch.arange(len(dataset) - 10))

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        self.trainer.predict_loop._return_predictions = True
        self.state_dict = deepcopy(self.trainer.lightning_module.state_dict())
        if self.trainer.datamodule is not None:
            self.trainer.datamodule.register_dataset_hooks(self._to_active_dataset, RunningStage.TRAINING)
            self.trainer.datamodule.register_dataset_hooks(self._to_active_dataset, RunningStage.VALIDATING)
            self.trainer.datamodule.register_dataset_hooks(self._to_active_dataset, RunningStage.TESTING)
            self.trainer.datamodule.register_dataset_hooks(self._to_active_dataset, RunningStage.PREDICTING)

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        self.trainer.train_dataloader = None
        self.trainer.val_dataloaders = None
        self.trainer.lightning_module.train_dataloader.unpatch(self.trainer.lightning_module)
        self.trainer.lightning_module.train_dataloader.unpatch(self.trainer.lightning_module)
        self.trainer.reset_train_val_dataloaders(self.trainer.lightning_module)
        self.trainer.reset_predict_dataloader(self.trainer.lightning_module)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self.progress.increment_started()
        self.fit_loop.run()
        self._reset_predicting()
        predictions = self.trainer.predict_loop.run()
        uncertainties = [self.heuristics.get_uncertainties(np.asarray(p)) for idx, p in enumerate(predictions)]
        _ = np.argsort(uncertainties)
        self._reset_fitting()
        self.progress.increment_processed()

    def on_advance_end(self) -> None:
        self.trainer.lightning_module.load_state_dict(self.state_dict)
        self.progress.increment_completed()
        return super().on_advance_end()

    def on_run_end(self):
        self.trainer.lightning_module.predict_step = self.trainer.lightning_module._predict_step
        return super().on_run_end()

    @property
    def done(self) -> bool:
        return self.progress.current.completed > self.max_epochs or self._should_continue

    def reset(self) -> None:
        pass

    @staticmethod
    def _mc_inference(*args, predict_step_fn: Callable = None, inference_iteration: int = None, **kwargs):
        out = []
        for _ in range(inference_iteration):
            out.append(predict_step_fn(*args, **kwargs))
        return out

    def _reset_fitting(self):
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.state.status = TrainerStatus.RUNNING
        self.trainer.training = True
        self.trainer.lightning_module.on_train_dataloader()

    def _identity(self, *args, **kwargs):
        pass

    def _reset_predicting(self):
        self.trainer.state.fn = TrainerFn.PREDICTING
        self.trainer.state.status = TrainerStatus.RUNNING
        self.trainer.predicting = True
        self.trainer.lightning_module.on_predict_dataloader()
        self.trainer.lightning_module.on_predict_model_eval = self._identity
        for _, module in self.trainer.lightning_module.named_modules():
            if isinstance(module, _BatchNorm):
                module.eval()
        if getattr(self.trainer.lightning_module, "_predict_step", None) is None:
            self.trainer.lightning_module._predict_step = self.trainer.lightning_module.predict_step
        self.trainer.lightning_module.predict_step = partial(
            self._mc_inference,
            predict_step_fn=self.trainer.lightning_module._predict_step,
            inference_iteration=self.inference_iteration,
        )

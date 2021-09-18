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
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus
from torch.nn.modules.batchnorm import _BatchNorm

import flash
from flash.core.utilities.imports import _BAAL_AVAILABLE, requires

if _BAAL_AVAILABLE:
    from baal.active.heuristics import AbstractHeuristic, BALD


class ActiveLearningTrainer(flash.Trainer):
    @requires("baal")
    def __init__(
        self,
        *args,
        val_size: Union[float, Callable] = 0,
        labelisation_stopping_criteria: Optional[Callable] = None,
        heuristic: "AbstractHeuristic" = BALD(reduction=np.mean),
        dataset_transform_fn: Optional[Callable] = None,
        label_epoch_frequency: int = 1,
        **kwags
    ):
        super().__init__(*args, **kwags)

        active_learning_loop = ActiveLearningLoop(
            dataset_transform_fn=dataset_transform_fn,
            val_size=val_size,
            heuristic=heuristic,
            labelisation_stopping_criteria=labelisation_stopping_criteria,
            label_epoch_frequency=label_epoch_frequency,
        )
        active_learning_loop.connect(self.fit_loop)
        self.fit_loop = active_learning_loop
        active_learning_loop.trainer = self


class ActiveLearningLoop(Loop):
    @requires("baal")
    def __init__(
        self,
        dataset_transform_fn: Optional[Callable],
        val_size: Union[float, Callable],
        heuristic: "AbstractHeuristic",
        labelisation_stopping_criteria: Optional[Callable],
        label_epoch_frequency: int = 1,
        inference_iteration: int = 2,
    ):
        super().__init__()
        self.label_epoch_frequency = label_epoch_frequency
        self.fit_loop: Optional[FitLoop] = None
        self.state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.progress = Progress()
        self.dataset_transform_fn = dataset_transform_fn
        self.val_size = val_size
        self.labelisation_stopping_criteria = labelisation_stopping_criteria
        self.heuristic = heuristic
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

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        self._reorder_datasets()
        self.trainer.predict_loop._return_predictions = True
        self.state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        self._merge_datasets()
        self.trainer.train_dataloader = None
        self.trainer.val_dataloaders = None
        self.trainer.lightning_module.train_dataloader.unpatch(self.trainer.lightning_module)
        self.trainer.lightning_module.train_dataloader.unpatch(self.trainer.lightning_module)
        self.trainer.reset_train_val_dataloaders(self.trainer.lightning_module)
        self.trainer.lightning_module.predict_dataloader = self.trainer.datamodule.predict_dataloader
        self.trainer.reset_predict_dataloader(self.trainer.lightning_module)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self.progress.increment_started()
        self.fit_loop.run()
        self._reset_predicting()
        predictions = self.trainer.predict_loop.run()
        uncertainties = [self.heuristic.get_uncertainties(np.asarray(p)) for idx, p in enumerate(predictions)]
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

    @property
    def has_training_data(self) -> bool:
        return self.trainer.datamodule._train_ds is not None

    def reset(self) -> None:
        pass

    def request_label(self):
        print("Requesting labellization.")

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

    def _merge_datasets(self) -> None:
        pass

    def _reorder_datasets(self) -> None:
        train_ds = self.trainer.datamodule._train_ds

        if isinstance(train_ds, list):
            for ds in train_ds:
                if ds.is_labelled:
                    self.trainer.datamodule.train_ds = ds
                else:
                    self.trainer.datamodule.predict_ds = ds

        if not self.has_training_data:
            self.request_label()

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

import torch
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerStatus
from torch.nn.modules.batchnorm import _BatchNorm

from flash.core.utilities.imports import requires
from flash.image.classification.integrations.baal.data import ActiveLearningDataModule


class ActiveLearningLoop(Loop):
    @requires("baal")
    def __init__(self, label_epoch_frequency: int, inference_iteration: int = 2):
        """The `ActiveLearning Loop` describes the following training procedure. This loop is connected with the
        `ActiveLearningTrainer`

        Example::

            while unlabelled data or budget critera not reached:

                if labelled data
                    trainer.fit(model, labelled data)

                if unlabelled data:
                    predictions = trainer.predict(model, unlabelled data)
                    uncertainties = heuristic(predictions)
                    request labellelisation for the sample with highest uncertainties under a given budget

        Args:
            label_epoch_frequency: Number of epoch to train on before requesting labellisation.
            inference_iteration: Number of inference to perform to compute uncertainty.
        """
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
        assert isinstance(self.trainer.datamodule, ActiveLearningDataModule)

    def reset(self) -> None:
        pass

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        self._reset_dataloader_for_stage(RunningStage.TRAINING)
        self._reset_dataloader_for_stage(RunningStage.PREDICTING)
        self.progress.increment_ready()

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self.progress.increment_started()
        self.fit_loop.run()
        if self.trainer.datamodule.has_unlabelled_data:
            self._reset_predicting()
            predictions = self.trainer.predict_loop.run()
            self.trainer.datamodule.label(predictions=predictions)
        self._reset_fitting()
        self.progress.increment_processed()

    def on_advance_end(self) -> None:
        if self.trainer.datamodule.has_unlabelled_data:
            # reload the weights to retrain from scratch with the new labelled data.
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

    ############## UTILS ##############

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

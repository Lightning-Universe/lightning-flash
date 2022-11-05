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
from typing import Any, Dict, List, Set, Tuple

from lightning_utilities.core.overrides import is_overridden

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.utils import _CALLBACK_FUNCS
from flash.core.utilities.stages import RunningStage


class BaseVisualization(BaseDataFetcher):
    """This Base Class is used to create visualization tool on top of
    :class:`~flash.core.data.io.input_transform.InputTransform` hooks.

    Override any of the ``show_{_hook_name}`` to receive the associated data and visualize them.

    Example::

        from flash.image import ImageClassificationData
        from flash.core.data.base_viz import BaseVisualization

        class CustomBaseVisualization(BaseVisualization):

            def show_load_sample(self, samples: List[Any], running_stage):
                # plot samples

            def show_per_sample_transform(self, samples: List[Any], running_stage):
                # plot samples

            def show_collate(self, batch: List[Any], running_stage):
                # plot batch

            def show_per_batch_transform(self, batch: List[Any], running_stage):
                # plot batch

        class CustomImageClassificationData(ImageClassificationData):

            @staticmethod
            def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
                return CustomBaseVisualization(*args, **kwargs)

        dm = CustomImageClassificationData.from_folders(
            train_folder="./data/train",
            val_folder="./data/val",
            test_folder="./data/test",
            predict_folder="./data/predict")

        # visualize a ``train`` batch
        dm.show_train_batches()

        # visualize next ``train`` batch
        dm.show_train_batches()

        # visualize a ``val`` batch
        dm.show_val_batches()

        # visualize a ``test`` batch
        dm.show_test_batches()

        # visualize a ``predict`` batch
        dm.show_predict_batches()

    .. note::

        If the user wants to plot all different transformation stages at once,
        override the ``show`` function directly.

    Example::

        class CustomBaseVisualization(BaseVisualization):

            def show(self, batch: Dict[str, Any], running_stage: RunningStage):
                print(batch)
                # out
                {
                    'load_sample': [...],
                    'per_sample_transform': [...],
                    'collate': [...],
                    'per_batch_transform': [...],
                }

    .. note::

        As the :class:`~flash.core.data.io.input_transform.InputTransform` hooks are injected within
        the threaded workers of the DataLoader,
        the data won't be accessible when using ``num_workers > 0``.
    """

    def _show(
        self,
        running_stage: RunningStage,
        func_names_list: List[str],
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        self.show(self.batches[running_stage], running_stage, func_names_list, limit_nb_samples, figsize)

    def show(
        self,
        batch: Dict[str, Any],
        running_stage: RunningStage,
        func_names_list: List[str],
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        """Override this function when you want to visualize a composition."""
        # filter out the functions to visualise
        func_names_set: Set[str] = set(func_names_list) & set(_CALLBACK_FUNCS)
        if len(func_names_set) == 0:
            raise ValueError(f"Invalid function names: {func_names_list}.")

        for func_name in func_names_set:
            hook_name = f"show_{func_name}"
            if is_overridden(hook_name, self, BaseVisualization):
                getattr(self, hook_name)(batch[func_name], running_stage, limit_nb_samples, figsize)

    def show_load_sample(
        self,
        samples: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        """Override to visualize  ``load_sample`` output data."""

    def show_per_sample_transform(
        self,
        samples: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        """Override to visualize ``per_sample_transform`` output data."""

    def show_collate(
        self,
        batch: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        """Override to visualize  ``collate`` output data."""

    def show_per_batch_transform(
        self,
        batch: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        """Override to visualize  ``per_batch_transform`` output data."""

    def show_per_sample_transform_on_device(
        self,
        samples: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        """Override to visualize  ``per_sample_transform_on_device`` output data."""

    def show_per_batch_transform_on_device(
        self,
        batch: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        """Override to visualize  ``per_batch_transform_on_device`` output data."""

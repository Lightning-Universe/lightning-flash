from typing import Any, Dict, List, Sequence

from pytorch_lightning.trainer.states import RunningStage

from flash.core.utils import _is_overriden
from flash.data.callback import BaseDataFetcher
from flash.data.utils import _PREPROCESS_FUNCS


class BaseViz(BaseDataFetcher):
    """
    This Base Class is used to create visualization tool on top of :class:`~flash.data.process.Preprocess` hooks.

    Override any of the ``show_{preprocess_hook_name}`` to receive the associated data and visualize them.

    Example::

        from flash.vision import ImageClassificationData
        from flash.data.base_viz import BaseViz

        class CustomBaseViz(BaseViz):

            def show_load_sample(self, samples: List[Any], running_stage):
                # plot samples

            def show_pre_tensor_transform(self, samples: List[Any], running_stage):
                # plot samples

            def show_to_tensor_transform(self, samples: List[Any], running_stage):
                # plot samples

            def show_post_tensor_transform(self, samples: List[Any], running_stage):
                # plot samples

            def show_collate(self, batch: List[Any], running_stage):
                # plot batch

            def show_per_batch_transform(self, batch: List[Any], running_stage):
                # plot batch

        class CustomImageClassificationData(ImageClassificationData):

            @staticmethod
            def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
                return CustomBaseViz(*args, **kwargs)

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

        class CustomBaseViz(BaseViz):

            def show(self, batch: Dict[str, Any], running_stage: RunningStage):
                print(batch)
                # out
                {
                    'load_sample': [...],
                    'pre_tensor_transform': [...],
                    'to_tensor_transform': [...],
                    'post_tensor_transform': [...],
                    'collate': [...],
                    'per_batch_transform': [...],
                }

    .. note::

        As the :class:`~flash.data.process.Preprocess` hooks are injected within
        the threaded workers for the DataLoader,
        the data won't be accessible when using ``num_workers > 0``.

    """

    def _show(self, running_stage: RunningStage) -> None:
        self.show(self.batches[running_stage], running_stage)

    def show(self, batch: Dict[str, Any], running_stage: RunningStage) -> None:
        """
        Override this function when you want to visualize a composition.
        """
        for func_name in _PREPROCESS_FUNCS:
            hook_name = f"show_{func_name}"
            if _is_overriden(hook_name, self, BaseViz):
                getattr(self, hook_name)(batch[func_name], running_stage)

    def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
        """Override to visualize preprocess ``load_sample`` output data."""
        pass

    def show_pre_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        """Override to visualize preprocess ``pre_tensor_transform`` output data."""
        pass

    def show_to_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        """Override to visualize preprocess ``to_tensor_transform`` output data."""
        pass

    def show_post_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        """Override to visualize preprocess ``post_tensor_transform`` output data."""
        pass

    def show_collate(self, batch: List[Any], running_stage: RunningStage) -> None:
        """Override to visualize preprocess ``collate`` output data."""
        pass

    def show_per_batch_transform(self, batch: List[Any], running_stage: RunningStage) -> None:
        """Override to visualize preprocess ``per_batch_transform`` output data."""
        pass

    def show_per_sample_transform_on_device(self, samples: List[Any], running_stage: RunningStage) -> None:
        """Override to visualize preprocess ``per_sample_transform_on_device`` output data."""
        pass

    def show_per_batch_transform_on_device(self, batch: List[Any], running_stage: RunningStage) -> None:
        """Override to visualize preprocess ``per_batch_transform_on_device`` output data."""
        pass

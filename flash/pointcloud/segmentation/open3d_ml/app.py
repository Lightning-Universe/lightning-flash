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
import torch

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _POINTCLOUD_AVAILABLE

if _POINTCLOUD_AVAILABLE:

    from open3d._ml3d.torch.dataloaders import TorchDataloader
    from open3d._ml3d.vis.visualizer import LabelLUT
    from open3d._ml3d.vis.visualizer import Visualizer as Open3dVisualizer

else:

    Open3dVisualizer = object


class Visualizer(Open3dVisualizer):
    def visualize_dataset(self, dataset, split, indices=None, width=1024, height=768):
        """Visualize a dataset.

        Example:
            Minimal example for visualizing a dataset::
                import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d

                dataset = ml3d.datasets.SemanticKITTI(dataset_path='/path/to/SemanticKITTI/')
                vis = ml3d.vis.Visualizer()
                vis.visualize_dataset(dataset, 'all', indices=range(100))

        Args:
            dataset: The dataset to use for visualization.
            split: The dataset split to be used, such as 'training'
            indices: An iterable with a subset of the data points to visualize, such as [0,2,3,4].
            width: The width of the visualization window.
            height: The height of the visualization window.
        """
        # Setup the labels
        lut = LabelLUT()
        color_map = dataset.color_map
        for id_color, val in dataset.label_to_names.items():
            lut.add_label(str(id_color), id_color, color=color_map[id_color])
        self.set_lut("labels", lut)

        self._consolidate_bounding_boxes = True
        self._init_dataset(dataset, split, indices)
        self._visualize("Open3D - " + dataset.name, width, height)


class App:
    def __init__(self, datamodule: DataModule):
        self.datamodule = datamodule
        self._enabled = True  # not flash._IS_TESTING

    def get_dataset(self, stage: str = "train"):
        dataloader = getattr(self.datamodule, f"{stage}_dataloader")()
        dataset = dataloader.dataset.dataset
        if isinstance(dataset, TorchDataloader):
            return dataset.dataset
        return dataset

    def show_train_dataset(self, indices=None):
        if self._enabled:
            dataset = self.get_dataset("train")
            viz = Visualizer()
            viz.visualize_dataset(dataset, "all", indices=indices)

    def show_predictions(self, predictions):
        if self._enabled:
            dataset = self.get_dataset("train")
            color_map = dataset.color_map

            predictions_visualizations = []
            for prediction_batch in predictions:
                for pred in prediction_batch:
                    predictions_visualizations.append(
                        {
                            "points": pred[DataKeys.INPUT],
                            "labels": pred[DataKeys.TARGET],
                            "predictions": torch.argmax(pred[DataKeys.PREDS], axis=-1) + 1,
                            "name": pred[DataKeys.METADATA]["name"],
                        }
                    )

            viz = Visualizer()
            lut = LabelLUT()
            color_map = dataset.color_map
            for id_color, val in dataset.label_to_names.items():
                lut.add_label(str(id_color), id_color, color=color_map[id_color])
            viz.set_lut("labels", lut)
            viz.set_lut("predictions", lut)
            viz.visualize(predictions_visualizations)


def launch_app(datamodule: DataModule) -> "App":
    return App(datamodule)

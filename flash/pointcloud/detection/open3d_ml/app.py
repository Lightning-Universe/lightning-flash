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
import numpy as np
from torch.utils.data.dataset import Dataset

import flash
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _POINTCLOUD_AVAILABLE

if _POINTCLOUD_AVAILABLE:

    from open3d._ml3d.vis.visualizer import LabelLUT, Visualizer
    from open3d.visualization import gui

    class Visualizer(Visualizer):
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
            for idx, color in dataset.color_map.items():
                lut.add_label(idx, idx, color=color)
            self.set_lut("label", lut)

            self._consolidate_bounding_boxes = True
            self._init_dataset(dataset, split, indices)

            self._visualize("Open3D - " + dataset.name, width, height)

        def _visualize(self, title, width, height):
            gui.Application.instance.initialize()
            self._init_user_interface(title, width, height)

            # override just to set background color to back :)
            bgcolor = gui.ColorEdit()
            bgcolor.color_value = gui.Color(0, 0, 0)
            self._on_bgcolor_changed(bgcolor.color_value)

            self._3d.scene.downsample_threshold = 400000

            # Turn all the objects off except the first one
            for name, node in self._name2treenode.items():
                node.checkbox.checked = False
                self._3d.scene.show_geometry(name, False)
            for name in [self._objects.data_names[0]]:
                self._name2treenode[name].checkbox.checked = True
                self._3d.scene.show_geometry(name, True)

            def on_done_ui():
                # Add bounding boxes here: bounding boxes belonging to the dataset
                # will not be loaded until now.
                self._update_bounding_boxes()

                self._update_Input_combobox()
                self._update_shaders_combobox()

                # Display "colors" by default if available, "points" if not
                available_attrs = self._get_available_attrs()
                self._set_shader(self.SOLID_NAME, force_update=True)
                if "colors" in available_attrs:
                    self._Input_combobox.selected_text = "colors"
                elif "points" in available_attrs:
                    self._Input_combobox.selected_text = "points"

                self._dont_update_geometry = True
                self._on_Input_changed(self._Input_combobox.selected_text, self._Input_combobox.selected_index)
                self._update_geometry_colors()
                self._dont_update_geometry = False
                # _Input_combobox was empty, now isn't, re-layout.
                self.window.set_needs_layout()

                self._update_geometry()
                self.setup_camera()

            self._load_geometries(self._objects.data_names, on_done_ui)
            gui.Application.instance.run()

    class VizDataset(Dataset):

        name = "VizDataset"

        def __init__(self, dataset):
            self.dataset = dataset
            self.label_to_names = getattr(dataset, "label_to_names", {})
            self.path_list = getattr(dataset, "path_list", [])
            self.color_map = getattr(dataset, "color_map", {})

        def get_data(self, index):
            data = self.dataset[index]["data"]
            data["bounding_boxes"] = data["bbox_objs"]
            data["color"] = np.ones_like(data["point"])
            return data

        def get_attr(self, index):
            return self.dataset[index]["attr"]

        def get_split(self, *_) -> "VizDataset":
            return self

        def __len__(self) -> int:
            return len(self.dataset)

    class App:
        def __init__(self, datamodule: DataModule):
            self.datamodule = datamodule
            self._enabled = not flash._IS_TESTING

        def get_dataset(self, stage: str = "train"):
            dataloader = getattr(self.datamodule, f"{stage}_dataloader")()
            return VizDataset(dataloader.dataset)

        def show_train_dataset(self, indices=None):
            if self._enabled:
                dataset = self.get_dataset("train")
                viz = Visualizer()
                viz.visualize_dataset(dataset, "all", indices=indices)

        def show_predictions(self, predictions):
            if self._enabled:
                dataset = self.get_dataset("train")

                viz = Visualizer()
                lut = LabelLUT()
                for id_color, color in dataset.color_map.items():
                    lut.add_label(str(id_color), id_color, color=color)
                viz.set_lut("label", lut)

                for prediction_batch in predictions:
                    for pred in prediction_batch:
                        data = {
                            "points": pred[DataKeys.INPUT][:, :3],
                            "name": pred[DataKeys.METADATA],
                        }
                        bounding_box = pred[DataKeys.PREDS]

                        viz.visualize([data], bounding_boxes=bounding_box)


def launch_app(datamodule: DataModule) -> "App":
    return App(datamodule)
